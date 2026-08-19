from agno.agent import Agent
from agno.models.dashscope import DashScope
from dotenv import load_dotenv
import os
import json

load_dotenv()


# =========================
# Underlying LLM Agent
# =========================
_llm_content_agent = Agent(
    name="content-analysis-agent",
    model=DashScope(
        id="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    description="""
        You are a professional public speaking content coach.
        You evaluate speech transcripts for structure, clarity, grammar, and filler words.
    """,
    instructions=[

        "You will receive a raw transcript string.",

        "Your job is to analyze it and return structured JSON.",

        "Return ONLY valid JSON in this exact structure:",

        "{",
        '  "grammar_corrections": [string],',
        '  "filler_words": {',
        '     "word": number',
        "  },",
        '  "structure_feedback": string,',
        '  "clarity_feedback": string,',
        '  "improvement_suggestions": [string]',
        "}",

        "Do NOT include markdown.",
        "Do NOT include explanations.",
        "Valid JSON only."
    ],
    markdown=False,
    debug_mode=True
)


# =========================
# Wrapper Class
# =========================
class ContentAnalysisAgent:

    def run(self, transcript: str):

        if not transcript or not isinstance(transcript, str):
            return {
                "grammar_corrections": [],
                "filler_words": {},
                "structure_feedback": "No transcript provided.",
                "clarity_feedback": "",
                "improvement_suggestions": []
            }

        prompt = transcript

        response = _llm_content_agent.run(prompt)

        if isinstance(response, str):
            try:
                return json.loads(response)
            except Exception:
                return {
                    "grammar_corrections": [],
                    "filler_words": {},
                    "structure_feedback": "Failed to parse model output.",
                    "clarity_feedback": "",
                    "improvement_suggestions": []
                }

        return response


# 实例化
content_analysis_agent = ContentAnalysisAgent()
