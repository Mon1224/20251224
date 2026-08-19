from agno.agent import Agent
from agno.models.dashscope import DashScope
from agents.tools.voice_analysis_tool import analyze_voice_attributes as voice_analysis_tool
from dotenv import load_dotenv
import os

load_dotenv()

# Define the voice analysis agent
voice_analysis_agent = Agent(
    name="voice-analysis-agent",
    model=DashScope(
        id="qwen3.7-max-2026-05-17",
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        enable_thinking=True,
    ),
    tools=[voice_analysis_tool],
    description="""
        You are a voice analysis agent that evaluates vocal attributes like clarity, intonation, and pace.
        You will return the transcribed text, speech rate, pitch variation, and volume consistency.
    """,
    instructions=[
        "You will be provided with an audio file of a person speaking.",
        "Your task is to analyze the vocal attributes in the audio to detect speech rate, pitch variation, and volume consistency.",
        "The response MUST be in the following JSON format:",
        "The response MUST be a single JSON object, not wrapped in an array or list.",
        "{",
            '"transcription": [transcription]',
            '"speech_rate_wpm": [speech_rate_wpm],',
            '"pitch_variation": [pitch_variation],',
            '"volume_consistency": [volume_consistency]',
        "}",
        "The response MUST be in proper JSON format with keys and values in double quotes.",
        "The final response MUST not include any other text or anything else other than the JSON response."
    ],
    markdown=True,
    debug_mode=False
)

# audio = "../../videos/my_video.mp4"
# prompt = f"Analyze vocal attributes in the audio file to detect speech rate, pitch variation, and volume consistency in the following audio: {audio}"
# voice_analysis_agent.print_response(prompt, stream=True)

# # Run agent and return the response as a variable
# response: RunResponse = voice_analysis_agent.run(prompt)
# # Print the response in markdown format
# pprint_run_response(response, markdown=True)
