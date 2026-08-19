import os
from .facial_expression_base import FacialExpressionAnalyzer
from agno.agent import Agent
from agno.models.dashscope import DashScope

class LLMFacialExpressionInterpreter(FacialExpressionAnalyzer):
    """
    基于大语言模型的表情分析（语义推断）
    """

    def __init__(self):
        self.agent = Agent(
            name="facial-expression-llm-agent",
            model=DashScope(
                id="qwen-plus",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            instructions=[
                "You are a public speaking coach.",
                "You will receive facial expression metrics in JSON.",
                "Explain engagement level and give improvement suggestions.",
                "Do NOT invent numbers. Only interpret given metrics."
            ]
        )

    def interpret(self, cv_result: dict) -> str:
        prompt = f"""
                Here are facial expression metrics extracted from a video:
                {cv_result}
                Provide a concise professional feedback.
                """
        return self.agent.run(prompt).content