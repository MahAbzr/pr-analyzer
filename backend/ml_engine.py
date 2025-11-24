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

    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 80):
        """
        Convert text → token IDs → sliding-window chunks.
        Returns list of token ID lists.
        Handles long sequences by tokenizing without triggering length warnings.
        """
        # Tokenize with truncation disabled and max_length set high to avoid warnings
        tokens = self.codebert_tokenizer.encode(
            text,
            add_special_tokens=False,
            return_tensors=None
        )

        chunks = []
        i = 0

        while i < len(tokens):
            chunk = tokens[i:i + chunk_size]
            chunks.append(chunk)
            i += (chunk_size - overlap)

        return chunks

    @staticmethod
    def mean_pool(last_hidden_state, attention_mask):
        """
        Mean pooling over token embeddings using attention mask.
        """
        mask = attention_mask.unsqueeze(-1)  # [B,L,1]
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def embed_code(self, code: str) -> np.ndarray:
        """
        Compute CodeBERT embedding for a code snippet with chunking support.
        Handles long code by splitting into chunks and averaging embeddings.
        Runs on CPU to avoid GPU memory pressure.
        """
        CHUNK_LEN = 400
        OVERLAP = 80
        MAX_LEN = 512  # CodeBERT max length (actual limit)

        # Split code into chunks
        token_chunks = self.chunk_text(code, CHUNK_LEN, OVERLAP)

        if len(token_chunks) == 0:
            # Empty code - return zero vector
            return np.zeros((1, 768), dtype=np.float32)

        chunk_vecs = []

        with torch.no_grad():
            for tokens in token_chunks:
                # Ensure chunk doesn't exceed max length
                if len(tokens) > MAX_LEN:
                    tokens = tokens[:MAX_LEN]

                # Create attention mask (1 for real tokens)
                attention_mask = [1] * len(tokens)

                # Convert to tensors
                input_ids = torch.tensor([tokens], dtype=torch.long)
                attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long)

                # Get embeddings
                outputs = self.codebert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask_tensor
                )

                # Mean pooling instead of just CLS token
                pooled = self.mean_pool(outputs.last_hidden_state, attention_mask_tensor)
                chunk_vecs.append(pooled.cpu().numpy())

        # Average over all chunks
        file_vec = np.mean(np.vstack(chunk_vecs), axis=0, dtype=np.float32)
        return file_vec.reshape(1, -1)

    @staticmethod
    def call_llm(prompt: str, max_new: int = 8192 * 2) -> str:
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
