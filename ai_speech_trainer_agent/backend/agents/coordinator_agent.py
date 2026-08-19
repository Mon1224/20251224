from backend.agents.facial_expression_agent import facial_expression_agent
from backend.agents.voice_analysis_agent import voice_analysis_agent
from backend.agents.content_analysis_agent import content_analysis_agent
from backend.agents.feedback_agent import feedback_agent
import json


# =========================
# Safe unwrap helper
# =========================
def unwrap_agent_result(run_result):
    """
    Convert Agno RunOutput / string JSON / dict
    into pure Python dict.
    """
    if run_result is None:
        return {}

    # Agno RunOutput 常见字段（不同版本可能不同）
    candidate = None
    for attr in ("output", "content", "result", "text", "final"):
        if hasattr(run_result, attr):
            candidate = getattr(run_result, attr)
            break

    data = candidate if candidate is not None else run_result

    if isinstance(data, dict):
        return data

    if isinstance(data, str):
        try:
            return json.loads(data)
        except Exception:
            print("⚠ Failed to parse JSON string from agent.")
            return {}

    # last resort
    try:
        return json.loads(str(data))
    except Exception:
        print("⚠ Unknown agent return type:", type(data))
        return {}


# =========================
# Coordinator (Stable Version)
# =========================
class CoordinatorAgent:

    def run(self, video_path: str):

        # -----------------------------
        # Step 1: Facial Expression
        # -----------------------------
        print("▶ Step 1: Running facial expression analysis...")
        facial_run = facial_expression_agent.run({"video_path": video_path})
        facial_result = unwrap_agent_result(facial_run)
        print("✅ Facial done")

        # -----------------------------
        # Step 2: Voice Analysis
        # -----------------------------
        print("▶ Step 2: Running voice analysis (Whisper + acoustic metrics)...")
        voice_run = voice_analysis_agent.run(video_path)  # voice agent expects path str
        voice_result = unwrap_agent_result(voice_run)
        print("✅ Voice done")

        transcript = voice_result.get("transcription", "")

        # -----------------------------
        # Step 3: Content Analysis
        # -----------------------------
        print("▶ Step 3: Running content analysis...")
        content_run = content_analysis_agent.run(transcript)  # content agent expects transcript str
        content_result = unwrap_agent_result(content_run)
        print("✅ Content done")

        # -----------------------------
        # Step 4: Feedback
        # -----------------------------
        print("▶ Step 4: Generating final feedback...")

        # ✅ IMPORTANT FIX:
        # feedback_agent 是自定义 FeedbackAgent（run 需要 3 个参数）
        feedback_run = feedback_agent.run(facial_result, voice_result, content_result)

        # feedback_run 可能是 dict / str / RunOutput（看你 feedback_agent 内部实现）
        feedback_result = unwrap_agent_result(feedback_run)
        print("✅ Feedback done")

        # -----------------------------
        # Extract strengths/weaknesses
        # -----------------------------
        strengths = feedback_result.get("strengths", [])
        weaknesses = feedback_result.get("weaknesses", [])
        suggestions = feedback_result.get("suggestions", [])

        # -----------------------------
        # Final Unified Output (Frontend-friendly)
        # -----------------------------
        return {
            # 下面四个用于调试/排查
            "facial_expression_agent": facial_result,
            "voice_analysis_agent": voice_result,
            "content_analysis_agent": content_result,
            "feedback_agent": feedback_result,

            # ✅ Home.py 右侧展示（来自 facial）
            "engagement_metrics": facial_result.get("engagement_metrics", {}),
            "emotion_timeline": facial_result.get("emotion_timeline", []),

            # ✅ Feedback.py 三列数据
            "strengths": strengths,
            "weaknesses": weaknesses,
            "suggestions": suggestions,

            # ✅ Feedback.py 评分雷达图（通常读 session_state.feedback_response 这个 JSON 字符串）
            "feedback_response": json.dumps(feedback_result, ensure_ascii=False),
        }


# 实例化
coordinator_agent = CoordinatorAgent()
