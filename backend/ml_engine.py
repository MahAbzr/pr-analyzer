import os
import re
from pathlib import Path

import google.generativeai as genai
import lightgbm as lgb
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

ROOT = Path(".")
MODELS = ROOT / "models"
LGBM_MODEL_PATH = MODELS / "lightgbm_risk_regressor.txt"


class MLEngine:
    def __init__(self):
        self.codebert_id = 'microsoft/codebert-base'

    def load_models(self):
        print(f"[INFO] Loading LightGBM model from {LGBM_MODEL_PATH}")
        self.lgbm_model = lgb.Booster(model_file=str(LGBM_MODEL_PATH))

        print(f"[INFO] Loading CodeBERT: {self.codebert_id} (CPU)")
        self.codebert_tokenizer = AutoTokenizer.from_pretrained(self.codebert_id)
        self.codebert_model = AutoModel.from_pretrained(self.codebert_id)  # CPU

    def embed_code(self, code: str) -> np.ndarray:
        """
        Compute CodeBERT [CLS] embedding for a code snippet.
        Runs on CPU to avoid GPU memory pressure.
        """
        inputs = self.codebert_tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self.codebert_model(**inputs)
            cls_vec = outputs.last_hidden_state[:, 0, :]
        return cls_vec.numpy().reshape(1, -1)

    @staticmethod
    def call_llm(prompt: str, max_new: int = 8192) -> str:
        """
        Call Mistral via HuggingFace Inference API using conversational task.
        Requires HUGGINGFACE_API_KEY environment variable.
        """
        api_key = os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": max_new,  # similar to max_new_tokens
                "top_p": 0.95,
            },
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
            ],
        )

        return model.generate_content(prompt).text

    @staticmethod
    def extract_fixed(raw: str) -> str:
        pattern = r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```"
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            return match.group(1).strip()

        return raw.strip()

    @staticmethod
    def build_prompt(code: str, pred_risk: float) -> str:
        """
        Build repair prompt including the LightGBM risk score.
        """
        lines = [
            "You are a secure code assistant.",
            f"Predicted security risk score (0–10): {pred_risk:.3f}",
            "",
            "Find the security risk(s) if they exist and provide a fixed version of the code.",
            "Provide ONLY the fixed code wrapped in triple backticks (```) with the language identifier.",
            "Do not include any explanations, comments, or text outside the code block.",
            "If there are no security issues, return the original code in the same format.",
            "",
            "Original code:",
            code,
            "",
            "Fixed code (wrapped in ``` with language identifier):",
        ]
        return "\n".join(lines)

    def analyze_security(self, code) -> tuple[float, float, str]:
        """
        Given code, get LLM suggestion and calculate risk scores before and after.
        Returns (pred_before, pred_after, fixed_code).
        """

        # load models
        self.load_models()

        emb_before = self.embed_code(code.strip())
        pred_before = float(self.lgbm_model.predict(emb_before)[0])

        prompt = self.build_prompt(code.strip(), pred_before)
        llm_output = self.call_llm(prompt)
        fixed_code = self.extract_fixed(llm_output)

        if fixed_code:
            emb_after = self.embed_code(fixed_code)
            pred_after = float(self.lgbm_model.predict(emb_after)[0])
        else:
            pred_after = pred_before  # no change

        return pred_before, pred_after, fixed_code


if __name__ == "__main__":
    sample_code = '''
def insecure_function(user_input):
    query = "SELECT * FROM users WHERE name = '" + user_input + "';"
    cursor.execute(query)
    '''
#     sample_code = '''
# def insecure_function(user_input):
#     query = "SELECT * FROM users WHERE name = ?;"
#     cursor.execute(query, (user_input,))
#     '''
    ml_engine = MLEngine()
    pred_b, pred_a, fixed = ml_engine.analyze_security(sample_code)
    print(f"Risk before: {pred_b:.3f}")
    print(f"Risk after: {pred_a:.3f}")
    print(f"\nFixed code:\n{fixed}")
