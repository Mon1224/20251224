from agno.agent import Agent
from agno.models.dashscope import DashScope
from backend.agents.tools.voice_analysis_tool import analyze_voice_attributes as voice_analysis_tool
from dotenv import load_dotenv
import os
import json

load_dotenv()


# =========================
# Underlying LLM Agent
# =========================
_llm_voice_agent = Agent(
    name="voice-analysis-agent",
    model=DashScope(
        id="qwen-flash-character",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    tools=[voice_analysis_tool],
    description="""
        You are a professional public speaking voice coach.
        You NEVER analyze raw audio yourself.
        You ALWAYS rely on the provided tool for transcription and acoustic metrics.
    """,
    instructions=[

        "You MUST call the analyze_voice_attributes tool.",
        "Do NOT manually infer speech metrics.",
        "The input will be a video file path.",

        "The tool returns transcription and acoustic metrics.",
        "Your job is to interpret those metrics professionally.",

        "Return ONLY valid JSON with the following structure:",

        "{",
        '  "transcript": string,',
        '  "acoustic_metrics": {',
        '    "speech_rate_wpm": number,',
        '    "pitch_variation": number,',
        '    "volume_consistency": number',
        "  },",
        '  "professional_feedback": {',
        '    "clarity": string,',
        '    "intonation": string,',
        '    "pace": string',
        "  }",
        "}",

        "Do NOT include markdown.",
        "Do NOT include explanations.",
        "Valid JSON only."
    ],
    markdown=False,
    debug_mode=True
)


# =========================
# Wrapper Class (Stable Entry)
# =========================
class VoiceAnalysisAgent:

    def run(self, video_path: str):

        # 构造明确 prompt，避免模型猜
        prompt = f"Analyze this video file: {video_path}"

        response = _llm_voice_agent.run(prompt)

        # agno 返回可能是字符串，需要转 dict
        if isinstance(response, str):
            try:
                return json.loads(response)
            except Exception:
                return {
                    "transcript": "",
                    "acoustic_metrics": {},
                    "professional_feedback": {
                        "clarity": "Failed to parse model output.",
                        "intonation": "",
                        "pace": ""
                    }
                }

        return response


# 实例化对外使用
voice_analysis_agent = VoiceAnalysisAgent()
