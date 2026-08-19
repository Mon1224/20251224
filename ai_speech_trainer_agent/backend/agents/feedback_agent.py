from agno.agent import Agent
from agno.models.dashscope import DashScope
from dotenv import load_dotenv
import os
import json

load_dotenv()


# =========================
# Underlying LLM Agent
# =========================
_llm_feedback_agent = Agent(
    name="feedback-agent",
    model=DashScope(
        id="qwen-flash-character",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    description="""
        You are a professional public speaking evaluation expert.
        You evaluate multi-modal analysis results and produce structured scoring feedback.
    """,
    instructions=[

        "You will receive structured JSON from three analysis modules:",
        "1) Facial expression analysis",
        "2) Voice analysis",
        "3) Content analysis",

        "You MUST evaluate the speaker based only on the provided structured data.",
        "Do NOT invent missing information.",

        "Score the speaker from 1 (Poor) to 5 (Excellent) on:",

        "1. Content & Organization",
        "2. Delivery & Vocal Quality",
        "3. Body Language & Eye Contact",
        "4. Audience Engagement",
        "5. Language & Clarity",

        "Then calculate total_score as the sum of all five scores (range 5-25).",

        "Interpretation must be exactly one of:",
        "- Needs significant improvement",
        "- Developing skills",
        "- Competent speaker",
        "- Proficient speaker",
        "- Outstanding speaker",

        "Return ONLY valid JSON in this exact structure:",

        "{",
        '  "scores": {',
        '    "content_organization": number,',
        '    "delivery_vocal_quality": number,',
        '    "body_language_eye_contact": number,',
        '    "audience_engagement": number,',
        '    "language_clarity": number',
        "  },",
        '  "total_score": number,',
        '  "interpretation": string,',
        '  "feedback_summary": string,',
        '  "strengths": [string],',
        '  "weaknesses": [string],',
        '  "suggestions": [string]',
        "}",

        "You MUST directly address the speaker using second person (e.g., 'You demonstrate...', 'You could improve...').",

        "No markdown.",
        "No explanations.",
        "Valid JSON only."
    ],
    markdown=False,
    debug_mode=True
)


# =========================
# Wrapper Class
# =========================
class FeedbackAgent:

    def run(self, facial_result: dict, voice_result: dict, content_result: dict):

        combined_input = {
            "facial_expression_analysis": facial_result,
            "voice_analysis": voice_result,
            "content_analysis": content_result
        }

        prompt = json.dumps(combined_input)

        response = _llm_feedback_agent.run(prompt)

        if isinstance(response, str):
            try:
                return json.loads(response)
            except Exception:
                return {
                    "scores": {},
                    "total_score": 0,
                    "interpretation": "Needs significant improvement",
                    "feedback_summary": "Failed to parse model output.",
                    "strengths": [],
                    "weaknesses": [],
                    "suggestions": []
                }

        return response


# 实例化
feedback_agent = FeedbackAgent()
