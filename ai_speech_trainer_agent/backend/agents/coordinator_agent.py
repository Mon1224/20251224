from agno.team.team import Team
from agno.models.dashscope import DashScope
from agents.facial_expression_agent import facial_expression_agent
from agents.voice_analysis_agent import voice_analysis_agent
from agents.content_analysis_agent import content_analysis_agent
from agents.feedback_agent import feedback_agent
import os
from pydantic import BaseModel, Field

# Structured response
class CoordinatorResponse(BaseModel):
    facial_expression_response: str = Field(..., description="Response from facial expression agent")
    voice_analysis_response: str = Field(..., description="Response from voice analysis agent")
    content_analysis_response: str = Field(..., description="Response from content analysis agent")
    feedback_response: str = Field(..., description="Response from feedback agent")
    strengths: list[str] = Field(..., description="List of speaker's strengths")
    weaknesses: list[str] = Field(..., description="List of speaker's weaknesses")
    suggestions: list[str] = Field(..., description="List of suggestions for speaker to improve")

# Initialize a team of agents
coordinator_agent = Team(
    name="coordinator-agent",
    model=DashScope(
        id="qwen3.7-max-2026-05-17",
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        enable_thinking=True,
    ),
    members=[facial_expression_agent, voice_analysis_agent, content_analysis_agent, feedback_agent],
    description="You are a public speaking coach who helps individuals improve their presentation skills through feedback and analysis.",
    instructions=[
        "You will be provided with a video file of a person speaking in a public setting.",
        "You will analyze the speaker's facial expressions, voice modulation, and content delivery to provide constructive feedback.",
        "Ask the facial expression agent to analyze the video file to detect emotions and engagement.",
        "Ask the voice analysis agent to analyze the audio file to detect speech rate, pitch variation, and volume consistency.",
        "Ask the content analysis agent to analyze the transcript given by voice analysis agent for structure, clarity, and filler words.",
        "If the voice analysis agent reports an error (its response contains an 'error' field or an empty transcription), do NOT ask the content analysis agent to analyze it; instead report that content analysis was skipped because the audio could not be transcribed.",
        "Ask the feedback agent to evaluate the analysis results from the facial expression agent, voice analysis agent, and content analysis agent to provide feedback on the overall effectiveness of the presentation, highlighting strengths and areas for improvement.",
        "Your response MUST include the exact responses from the facial expression agent, voice analysis agent, content analysis agent, and feedback agent.",
        "Additionally, your response MUST provide lists of the speaker's strengths, weaknesses, and suggestions for improvement based on all the responses and feedback provided by the feedback agent.",
        "Important: You MUST directly address the speaker while providing strengths, weaknesses, and suggestions for improvement in a clear and constructive manner.",
        "The response MUST be in the following strict JSON format:",
        "The response MUST be a single JSON object. It MUST NOT be wrapped in an array or list, and MUST NOT contain any text outside the object.",
        "{",
            '"facial_expression_response": "<the facial expression agent response as a string>",',
            '"voice_analysis_response": "<the voice analysis agent response as a string>",',
            '"content_analysis_response": "<the content analysis agent response as a string>",',
            '"feedback_response": "<the feedback agent response as a string>",',
            '"strengths": ["<strength 1>", "<strength 2>"],',
            '"weaknesses": ["<weakness 1>", "<weakness 2>"],',
            '"suggestions": ["<suggestion 1>", "<suggestion 2>"]',
        "}",
        "The response MUST start with '{' and end with '}'. Do NOT wrap the JSON object in square brackets.",
        "The response MUST be in strict JSON format with keys and values in double quotes.",
        "The values in the JSON response MUST NOT be null or empty.",
        "The final response MUST not include any other text or anything else other than the JSON response.",
        "The final response MUST not include any backslashes in the JSON response.",
        "The final response MUST be a valid JSON object and MUST not have any unterminated strings in the JSON response."
    ],
    add_datetime_to_context=True,
    add_member_tools_to_context=False,  # This can be tried to make the agent more consistently get the transfer tool call correct
    enable_agentic_state=True,  # Allow the agent to maintain a shared context and send that to members.
    share_member_interactions=True,  # Share all member responses with subsequent member requests.
    show_members_responses=True,
    store_member_responses=True,  # Keep each member's raw response for the backend to use.
    markdown=True,
    debug_mode=False
)

# # Example usage
# video = "../../videos/my_video.mp4"
# prompt = f"Analyze facial expressions, voice modulation, and content delivery in the following video: {video} and provide constructive feedback."
# coordinator_agent.print_response(prompt, stream=True)

# # Run agent and return the response as a variable
# response: RunResponse = coordinator_agent.run(prompt)
# # Print the response in markdown format
# pprint_run_response(response, markdown=True)
