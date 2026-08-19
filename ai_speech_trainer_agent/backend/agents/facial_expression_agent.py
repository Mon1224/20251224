from agno.agent import Agent
from agno.models.dashscope import DashScope
from dotenv import load_dotenv
import os
from pydantic import BaseModel

from backend.agents.tools.facial_expression_tool import analyze_facial_expressions

load_dotenv()


# =========================
# Structured Input Schema
# =========================
class FacialExpressionInput(BaseModel):
    video_path: str


# =========================
# Facial Expression Agent
# =========================
facial_expression_agent = Agent(
    name="facial-expression-agent",

    model=DashScope(
        id="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),

    # ✅ 强制结构化输入
    input_schema=FacialExpressionInput,

    # ✅ 绑定工具
    tools=[analyze_facial_expressions],

    description="""
You are a facial expression analysis agent.
You MUST call the analyze_facial_expressions tool
using the structured input field 'video_path'.
""",

    instructions=[

        # 输入说明
        "You will receive input as a structured JSON object.",
        "The JSON contains a field called 'video_path'.",

        # 强制调用工具
        "You MUST call the tool analyze_facial_expressions.",
        "Pass the EXACT value of video_path to the tool.",
        "Do NOT modify the path.",
        "Do NOT fabricate results.",

        # 输出要求
        "After receiving the tool output, return it directly.",
        "Return ONLY valid JSON.",
        "Do NOT include markdown.",
        "Do NOT include explanations.",

        "The JSON must follow this structure exactly:",

        "{",
        '  "emotion_timeline": [',
        '    {"timestamp": number, "emotion": string}',
        "  ],",
        '  "engagement_metrics": {',
        '    "eye_contact_frequency": number,',
        '    "smile_frequency": number',
        "  }",
        "}",

        "All keys and string values must use double quotes."
    ],

    markdown=False,
    debug_mode=True
)
