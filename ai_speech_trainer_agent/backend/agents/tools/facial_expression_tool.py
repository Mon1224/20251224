from agno.tools import tool
from .facial_expression_core import analyze_facial_expressions_core


@tool(
    name="analyze_facial_expressions",
    description="Analyze facial expressions via external CV service and interpret with LLM",
    show_result=True
)
def analyze_facial_expressions(video_path: str) -> dict:
    return analyze_facial_expressions_core(video_path)