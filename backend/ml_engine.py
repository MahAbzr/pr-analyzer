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
    def extract_issues_and_hints(raw: str) -> tuple[str, str]:
        """
        Extract potential issues and hints from LLM response.
        Expected format:
        POTENTIAL ISSUES:
        - issue 1
        - issue 2

        HINTS:
        - hint 1
        - hint 2
        """
        issues = ""
        hints = ""

        # Try to extract structured sections
        issues_match = re.search(r"POTENTIAL\s+ISSUES?:?\s*\n(.*?)(?=\n\s*HINTS?:|\Z)", raw, re.DOTALL | re.IGNORECASE)
        hints_match = re.search(r"HINTS?:?\s*\n(.*?)(?=\n\s*POTENTIAL|\Z)", raw, re.DOTALL | re.IGNORECASE)

        if issues_match:
            issues = issues_match.group(1).strip()
        if hints_match:
            hints = hints_match.group(1).strip()

        # If structured format not found, try to split the response
        if not issues and not hints:
            parts = raw.split('\n\n', 1)
            if len(parts) == 2:
                issues = parts[0].strip()
                hints = parts[1].strip()
            else:
                # All content goes to issues
                issues = raw.strip()
                hints = "No specific hints provided."

        return issues, hints

    @staticmethod
    def build_prompt(code: str, pred_risk: float) -> str:
        """
        Build analysis prompt to identify potential security issues and provide hints.
        Requests max 5 short, accurate bullet points for each section.
        """
        lines = [
            "You are a security code analysis assistant.",
            f"The code has a predicted security risk score of {pred_risk:.2f} out of 10.",
            "",
            "Analyze the following code for security vulnerabilities and provide your response in this EXACT format:",
            "",
            "POTENTIAL ISSUES:",
            "- List the TOP 5 (or fewer) most critical security issues found",
            "- MUST include the line number(s) where each issue occurs (e.g., 'Line 3: SQL injection risk')",
            "- Each bullet point must be ONE short, clear sentence (max 15 words)",
            "- Be specific and accurate about the vulnerability",
            "- ONLY mention issues that actually exist in the code - DO NOT hallucinate or make up issues",
            "- If no issues found, write: 'No security issues detected'",
            "",
            "HINTS:",
            "- Provide the TOP 5 (or fewer) most important best practice recommendations",
            "- Each bullet point must be ONE short, actionable sentence (max 15 words)",
            "- Focus on concrete fixes, not general advice",
            "- Be direct and practical",
            "",
            "IMPORTANT: ",
            "- Use bullet points (-) only. Keep each point concise and under 15 words.",
            "- Always include line numbers for POTENTIAL ISSUES.",
            "- Only report real issues found in the actual code provided.",
            "",
            "Code to analyze:",
            code,
        ]
        return "\n".join(lines)

    def analyze_security(self, code: str) -> tuple[float, str, str]:
        """
        Analyze code for security issues.
        Returns (security_score, potential_issues, hints).
        """
        # load models
        self.load_models()

        emb = self.embed_code(code.strip())
        security_score = float(self.lgbm_model.predict(emb)[0])

        prompt = self.build_prompt(code.strip(), security_score)
        llm_output = self.call_llm(prompt)
        potential_issues, hints = self.extract_issues_and_hints(llm_output)

        return security_score, potential_issues, hints


if __name__ == "__main__":
    sample_code = '''
def insecure_function(user_input):
    query = "SELECT * FROM users WHERE name = '" + user_input + "';"
    cursor.execute(query)
    '''
    ml_engine = MLEngine()
    score, issues, hints = ml_engine.analyze_security(sample_code)
    print(f"Security Score: {score:.3f}/10")
    print(f"\nPotential Issues:\n{issues}")
    print(f"\nHints:\n{hints}")
