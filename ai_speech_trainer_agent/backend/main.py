from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import traceback

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ 接收前端的 video_url（实际就是本地临时路径）
class AnalysisRequest(BaseModel):
    video_url: str


@app.get("/")
async def root():
    return {"message": "Welcome to AI Speech Trainer API"}


def _to_plain(obj):
    """
    把 Agno 的 RunOutput / 其它不可序列化对象，尽量转成可 JSON 化的结构。
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(x) for x in obj]

    # Agno 的 RunOutput / pydantic 等
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass

    # 最后兜底：转字符串
    return str(obj)


@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    print("🟢 Received analyze request")
    print("Video path:", request.video_url)

    try:
        # ✅ 统一调度：直接跑 coordinator（五个 agent 串起来）
        from backend.agents.coordinator_agent import coordinator_agent

        result = coordinator_agent.run(request.video_url)
        result = _to_plain(result)  # 保证可序列化（即便里面混入 RunOutput）

        # -------------------------
        # 1) Home.py 右侧展示用（facial）
        # -------------------------
        engagement_metrics = result.get("engagement_metrics", {}) or {}
        emotion_timeline = result.get("emotion_timeline", []) or []

        # -------------------------
        # 2) Feedback 页面用：统一成 JSON 字符串（避免前端 json.loads 类型不匹配）
        # -------------------------
        voice_obj = result.get("voice_analysis_agent", {}) or {}
        content_obj = result.get("content_analysis_agent", {}) or {}

        voice_analysis_response = json.dumps(voice_obj, ensure_ascii=False)
        content_analysis_response = json.dumps(content_obj, ensure_ascii=False)

        # ✅ 优先使用 coordinator 已经准备好的 feedback_response（如果有）
        feedback_response = result.get("feedback_response", None)

        # fallback：如果 coordinator 没给 feedback_response，就从 feedback_agent 里拿再 dump
        if not feedback_response:
            feedback_obj = result.get("feedback_agent", {}) or {}
            if isinstance(feedback_obj, str):
                try:
                    feedback_obj = json.loads(feedback_obj)
                except Exception:
                    feedback_obj = {}
            feedback_response = json.dumps(feedback_obj, ensure_ascii=False)

        payload = {
            # ✅ Home.py 右侧用
            "engagement_metrics": engagement_metrics,
            "emotion_timeline": emotion_timeline,

            # ✅ Feedback.py 里通常会 json.loads(...) 再取字段
            "voice_analysis_response": voice_analysis_response,
            "content_analysis_response": content_analysis_response,
            "feedback_response": feedback_response,

            # ✅ Home.py 也会用到
            "strengths": result.get("strengths", []) or [],
            "weaknesses": result.get("weaknesses", []) or [],
            "suggestions": result.get("suggestions", []) or [],

            # （可选）保留原始输出，便于排错对照
            "raw": result,
        }

        return JSONResponse(content=payload)

    except Exception as e:
        print("🔴 Backend Error:", e)
        traceback.print_exc()
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
