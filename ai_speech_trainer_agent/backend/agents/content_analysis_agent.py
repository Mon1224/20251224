from agno.agent import Agent
from agno.models.dashscope import DashScope
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()


# Structured output: the content agent has no tools, so agno's JSON mode is
# safe here and keeps the response format stable.
class ContentAnalysisResult(BaseModel):
    grammar_corrections: list[str] = Field(default_factory=list)
    filler_words: dict[str, int] = Field(default_factory=dict)
    suggestions: list[str] = Field(default_factory=list)


# Define the content analysis agent
content_analysis_agent = Agent(
    name="content-analysis-agent",
    model=DashScope(
        id="qwen3.7-max-2026-05-17",
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        enable_thinking=True,
    ),
    description="""
        You are a content analysis agent that evaluates transcribed speech for structure, clarity, and filler words.
        You will return grammar corrections, identified filler words, and suggestions for content improvement.
    """,
    instructions=[
        "You will be provided with a transcript of spoken content.",
        "Your task is to analyze the transcript and identify:",
        "- Grammar and syntax corrections.",
        "- Filler words and their frequency.",
        "- Suggestions for improving clarity and structure.",
        "The response MUST be in the following JSON format:",
        "The response MUST be a single JSON object, not wrapped in an array or list.",
        "{",
            '"grammar_corrections": ["correction 1", "correction 2"],',
            '"filler_words": { "word": count },',
            '"suggestions": ["suggestion 1", "suggestion 2"]',
        "}",
        "Ensure the response is in proper JSON format with keys and values in double quotes.",
        "Do not include any additional text outside the JSON response."
    ],
    markdown=True,
    output_schema=ContentAnalysisResult,
    use_json_mode=True,
    debug_mode=False
)

# # Example usage
# if __name__ == "__main__":
#     # Sample transcript from the Voice Analysis Agent
#     transcript = (
#         "So, um, I was thinking that, like, we could actually start the project soon. "
#         "You know, it's basically ready, and, uh, we just need to finalize some details."
#     )
#     prompt = f"Analyze the following transcript:\n\n{transcript}"
#     content_analysis_agent.print_response(prompt, stream=True)

    # # Run agent and return the response as a variable
    # response: RunResponse = content_analysis_agent.run(prompt)
    # # Print the response in markdown format
    # pprint_run_response(response, markdown=True)
