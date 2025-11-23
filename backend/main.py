import os
import traceback
import uuid
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from requests import Request
from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from starlette.responses import JSONResponse

from ml_engine import MLEngine

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/code_analyzer")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
MLEngine = MLEngine()


# Database model
class CodeAnalysis(Base):
    __tablename__ = "code_analyses"

    id = Column(String, primary_key=True, index=True)
    original_code = Column(String)
    fixed_code = Column(String)
    before_score = Column(Float)
    after_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


# Request/Response models
class AnalyzeRequest(BaseModel):
    code_snippet: str


class AnalysisResult(BaseModel):
    id: str
    original_code: str
    fixed_code: str
    before_score: float
    after_score: float
    created_at: datetime

    class Config:
        from_attributes = True


# FastAPI app
app = FastAPI(title="Code Security Analyzer")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    print("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ============ Routes ============

@app.post("/api/analyze", response_model=AnalysisResult)
def analyze_code(request: AnalyzeRequest):
    """Analyze code snippet using ML Engine and save results"""
    db = SessionLocal()
    try:

        before_score, after_score, fixed_code = MLEngine.analyze_security(request.code_snippet)
        # Generate unique ID
        analysis_id = str(uuid.uuid4())

        if before_score < after_score:
            fixed_code = request.code_snippet  # No improvement, keep original
            after_score = before_score

        # Save to database
        analysis = CodeAnalysis(
            id=analysis_id,
            original_code=request.code_snippet,
            fixed_code=fixed_code,
            before_score=before_score,
            after_score=after_score
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        return AnalysisResult.model_validate(analysis)
    except Exception as e:
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/api/analysis/{analysis_id}", response_model=AnalysisResult)
def get_analysis(analysis_id: str):
    """Retrieve a specific analysis by ID"""
    db = SessionLocal()
    try:
        analysis = db.query(CodeAnalysis).filter(CodeAnalysis.id == analysis_id).first()
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return analysis
    finally:
        db.close()


@app.get("/api/analyses", response_model=list[AnalysisResult])
def get_all_analyses(skip: int = 0, limit: int = 10):
    """Get all analyses with pagination"""
    db = SessionLocal()
    try:
        analyses = db.query(CodeAnalysis).offset(skip).limit(limit).all()
        return analyses
    finally:
        db.close()


@app.delete("/api/analysis/{analysis_id}")
def delete_analysis(analysis_id: str):
    """Delete an analysis"""
    db = SessionLocal()
    try:
        analysis = db.query(CodeAnalysis).filter(CodeAnalysis.id == analysis_id).first()
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        db.delete(analysis)
        db.commit()
        return {"message": "Analysis deleted successfully"}
    finally:
        db.close()


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
