from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import httpx
import json
from datetime import datetime
import asyncio
import re
from sqlalchemy import create_engine, Column, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os
from dotenv import load_dotenv

load_dotenv()

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/code_analyzer")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# CHANGED: Added pr_metadata column
class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    id = Column(String, primary_key=True)
    url = Column(String)
    code_snippet = Column(String)
    extracted_data = Column(JSON)
    ml_result = Column(JSON)
    pr_metadata = Column(JSON)  # NEW: Store PR metadata
    status = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# CHANGED: Added pr_metadata field
class CodeInput(BaseModel):
    url: Optional[str] = None
    code: Optional[str] = None


class AnalysisResponse(BaseModel):
    id: str
    status: str
    extracted_data: Optional[dict] = None
    ml_result: Optional[dict] = None
    pr_metadata: Optional[dict] = None  # NEW: Include metadata in response
    message: str


# Initialize FastAPI app
app = FastAPI(title="Code Analyzer API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Data Extraction Service
class DataExtractionService:
    @staticmethod
    def extract_from_url(url: str) -> dict:
        """Extract code from GitHub PR or repository URL"""
        try:
            if "github.com" in url:
                # Convert GitHub URL to raw content URL
                if "/pull/" in url:
                    # PR URL
                    parts = url.replace("https://github.com/", "").split("/")
                    owner, repo, _, pr_num = parts[0], parts[1], parts[2], parts[3]
                    raw_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}"
                else:
                    # Raw file URL
                    raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

                response = httpx.get(raw_url, timeout=10)
                response.raise_for_status()
                content = response.text if not url.endswith(".json") else response.json()
                return {"source": "github", "content": content, "url": url}
        except Exception as e:
            raise ValueError(f"Failed to extract from URL: {str(e)}")

    @staticmethod
    def extract_code_features(code: str) -> dict:
        """Extract features from code snippet"""
        features = {
            "languages": DataExtractionService._detect_languages(code),
            "functions": DataExtractionService._extract_functions(code),
            "imports": DataExtractionService._extract_imports(code),
            "complexity_indicators": DataExtractionService._analyze_complexity(code),
            "code_length": len(code.split('\n')),
            "has_tests": bool(re.search(r'(test_|_test\.py|describe\(|it\()', code))
        }
        return features

    @staticmethod
    def _detect_languages(code: str) -> list:
        """Detect programming languages in code"""
        languages = []
        patterns = {
            "python": (r'import\s+\w+|from\s+\w+\s+import|def\s+\w+', "python"),
            "javascript": (r'const\s+\w+|function\s+\w+|async\s+function', "javascript"),
            "java": (r'public\s+class|import\s+java\.', "java"),
            "csharp": (r'using\s+System|public\s+class', "csharp"),
        }
        for lang, (pattern, name) in patterns.items():
            if re.search(pattern, code):
                languages.append(name)
        return languages if languages else ["unknown"]

    @staticmethod
    def _extract_functions(code: str) -> list:
        """Extract function/method names"""
        functions = re.findall(r'(?:def|function|async\s+function)\s+(\w+)', code)
        return list(set(functions))[:10]

    @staticmethod
    def _extract_imports(code: str) -> list:
        """Extract imports and dependencies"""
        imports = re.findall(r'(?:import|from)\s+([a-zA-Z0-9_\.]+)', code)
        return list(set(imports))[:15]

    @staticmethod
    def _analyze_complexity(code: str) -> dict:
        """Analyze code complexity indicators"""
        lines = code.split('\n')
        return {
            "nested_depth": max([len(re.match(r'^(\s*)', line).group(1)) // 4 for line in lines] or [0]),
            "has_loops": bool(re.search(r'\b(for|while)\b', code)),
            "has_conditionals": bool(re.search(r'\b(if|else|switch)\b', code)),
            "comment_ratio": len(re.findall(r'#|//', code)) / max(len(lines), 1)
        }


# ML Engine Service (mock)
class MLEngineService:
    @staticmethod
    async def analyze(extracted_data: dict) -> dict:
        """Send to ML engine and get results"""
        # This simulates ML engine analysis
        await asyncio.sleep(2)  # Simulate processing time

        result = {
            "quality_score": 0.80,
            "maintainability": 0.78,
            "complexity_level": "medium",
            "issues": [
                "High nested depth detected",
                "Missing error handling in 2 functions"
            ],
            "suggestions": [
                "Refactor nested functions",
                "Add try-catch blocks",
                "Add more unit tests"
            ],
            "risk_level": "low"
        }
        return result

# CHANGED: Added PR metadata extraction
@app.post("/api/analyze")
async def analyze_code(input_data: CodeInput, background_tasks: BackgroundTasks):
    """Analyze code from URL or direct input"""
    import uuid

    analysis_id = str(uuid.uuid4())
    db = SessionLocal()

    try:
        code_content = ""
        pr_metadata = {}

        # Extract code
        if input_data.url:
            # Extract metadata and code from GitHub PR
            code_files = []

            if "github.com" in input_data.url and "/pull/" in input_data.url:
                # This is a GitHub PR URL
                try:
                    # Parse PR URL
                    parts = input_data.url.replace("https://github.com/", "").split("/")
                    owner, repo, _, pr_num = parts[0], parts[1], parts[2], parts[3]

                    # Get PR details from GitHub API
                    pr_api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}"
                    pr_response = httpx.get(pr_api_url, timeout=10)
                    pr_response.raise_for_status()
                    pr_data = pr_response.json()

                    # Extract metadata
                    pr_metadata = {
                        "title": pr_data.get("title", ""),
                        "body": pr_data.get("body", ""),
                        "author": pr_data.get("user", {}).get("login", ""),
                        "created_at": pr_data.get("created_at", ""),
                        "updated_at": pr_data.get("updated_at", ""),
                        "state": pr_data.get("state", ""),
                        "additions": pr_data.get("additions", 0),
                        "deletions": pr_data.get("deletions", 0),
                        "changed_files": pr_data.get("changed_files", 0),
                    }

                    # Get PR files (changed code)
                    files_api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}/files"
                    files_response = httpx.get(files_api_url, timeout=10)
                    files_response.raise_for_status()
                    files_data = files_response.json()

                    # Extract code from each file
                    for file_info in files_data:
                        if file_info.get("status") != "removed":  # Skip deleted files
                            file_name = file_info.get("filename", "")
                            patch = file_info.get("patch", "")

                            code_files.append({
                                "filename": file_name,
                                "status": file_info.get("status", ""),
                                "additions": file_info.get("additions", 0),
                                "deletions": file_info.get("deletions", 0),
                                "changes": file_info.get("changes", 0),
                                "patch": patch
                            })

                    # Combine all code from files
                    combined_code = "\n".join([f"// File: {f['filename']}\n{f['patch']}" for f in code_files])
                    code_content = combined_code if combined_code else pr_data.get("body", "")

                except Exception as e:
                    raise ValueError(f"Failed to extract PR metadata: {str(e)}")
            else:
                # For non-PR URLs, extract normally
                extracted = DataExtractionService.extract_from_url(input_data.url)
                code_content = extracted["content"]

        elif input_data.code:
            code_content = input_data.code
        else:
            raise HTTPException(status_code=400, detail="Provide either URL or code")

        # Extract features
        extracted_features = DataExtractionService.extract_code_features(code_content)

        # Save to database
        analysis = AnalysisResult(
            id=analysis_id,
            url=input_data.url or "direct_input",
            code_snippet=code_content[:1000],
            extracted_data=extracted_features,
            pr_metadata=pr_metadata if pr_metadata else None,  # Store metadata
            status="processing",
            ml_result=None
        )
        db.add(analysis)
        db.commit()

        # Run ML analysis in background
        background_tasks.add_task(process_ml_analysis, analysis_id)

        return AnalysisResponse(
            id=analysis_id,
            status="processing",
            extracted_data=extracted_features,
            pr_metadata=pr_metadata if pr_metadata else None,  # Return metadata
            message="Analysis started. Check status for updates."
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()


# CHANGED: Added pr_metadata to response
@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get analysis results including PR metadata"""
    db = SessionLocal()
    analysis = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
    db.close()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return AnalysisResponse(
        id=analysis.id,
        status=analysis.status,
        extracted_data=analysis.extracted_data,
        ml_result=analysis.ml_result,
        pr_metadata=analysis.pr_metadata,  # Include metadata in response
        message="Analysis completed" if analysis.status == "completed" else "Processing..."
    )


async def process_ml_analysis(analysis_id: str):
    """Background task to process ML analysis"""
    db = SessionLocal()
    analysis = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()

    if analysis:
        try:
            ml_result = await MLEngineService.analyze(analysis.extracted_data)
            analysis.ml_result = ml_result
            analysis.status = "completed"
            db.commit()
        except Exception as e:
            analysis.status = "failed"
            analysis.ml_result = {"error": str(e)}
            db.commit()

    db.close()


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "Code Analyzer API"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)