from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from app.cv_engine import FacialExpressionCVEngine
import traceback
from fastapi.responses import JSONResponse

app = FastAPI()

# 全局只保留一个引用，但不立刻创建
_engine: Optional[FacialExpressionCVEngine] = None


def get_engine() -> FacialExpressionCVEngine:
    global _engine
    if _engine is None:
        print("[CV] Initializing FacialExpressionCVEngine...")
        _engine = FacialExpressionCVEngine()
        print("[CV] FacialExpressionCVEngine ready")
    return _engine


class AnalyzeRequest(BaseModel):
    video_path: str


@app.get("/")
def health():
    return {"status": "cv service alive"}


@app.post("/analyze")
def analyze_face(req: AnalyzeRequest):
    try:
        engine = get_engine()
        result = engine.analyze(req.video_path)
        return result
    except Exception as e:
        print("🔥 [CV] analyze_face crashed")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "video_path": req.video_path
            }
        )