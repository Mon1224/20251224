import os
from pathlib import Path

from dotenv import load_dotenv

# Load the .env file from the project root (backend/main.py -> project root),
# regardless of the current working directory. This must happen BEFORE importing
# the agents, so they pick up the API key when they construct their models.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agno.agent import RunOutput
from agents.coordinator_agent import coordinator_agent
import json
import re

# The API key used by the analysis agents. Read it here so a missing key is
# detected early and reported clearly instead of failing mid-analysis.
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
if not QWEN_API_KEY:
    print("WARNING: QWEN_API_KEY is not set. Add it to the .env file before running an analysis.")


def _extract_json_value(text: str):
    """Best-effort extraction of a JSON value from model output text."""
    candidates = [text.strip()]

    # Markdown code fence
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())

    # Balanced object span { ... }
    start = text.find("{")
    if start != -1:
        end = text.rfind("}")
        if end > start:
            candidates.append(text[start:end + 1])

    # Balanced array span [ ... ] (Qwen sometimes wraps the object in an array)
    start = text.find("[")
    if start != -1:
        end = text.rfind("]")
        if end > start:
            candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


# Initialize FastAPI app
app = FastAPI()

# Configure CORS to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    # Explicit origins only. Streamlit's default local address is allowed;
    # update this list if you run Streamlit on another port or host.
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request body model
class AnalysisRequest(BaseModel):
    video_url: str

# Define the entry point
@app.get("/")
async def root():
    return {"message": "Welcome to the video analysis API!"}

# Define the analysis endpoint
@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    # Normalize the path: forward slashes survive the trip through the LLM/JSON
    # tool call without backslash-escape corruption, and Windows APIs accept them.
    video_url = (request.video_url or "").strip().strip('"').replace("\\", "/")

    # The frontend stores uploaded videos in the shared temp directory, so the
    # file must already exist here. Fail fast with a clear error otherwise.
    if not video_url:
        return JSONResponse(status_code=400, content={"detail": "video_url is required"})
    if not os.path.exists(video_url):
        return JSONResponse(
            status_code=404,
            content={"detail": f"Video file not found: {video_url}"},
        )
    if not QWEN_API_KEY:
        return JSONResponse(
            status_code=503,
            content={"detail": "QWEN_API_KEY is missing. Add it to the .env file and restart the backend."},
        )

    prompt = f"Analyze the following video: {video_url}"

    # The agent run can fail for many reasons (LLM outage, JSON parse failure,
    # a tool exception, ...). Surface a clear error instead of a bare 500.
    try:
        response: RunOutput = coordinator_agent.run(prompt)
    except Exception as e:
        return JSONResponse(status_code=502, content={"detail": f"Analysis failed: {e}"})

    if response is None or response.content is None:
        return JSONResponse(status_code=502, content={"detail": "Analysis returned no result."})

    content = response.content

    # Without an output_schema on the team, content is the raw model text.
    # Extract JSON from it (markdown fences, extra text, or an array-wrapped
    # object are all handled); otherwise fail clearly instead of returning a
    # JSON-encoded string that breaks the frontend.
    if isinstance(content, str):
        parsed = _extract_json_value(content)
        if parsed is None:
            return JSONResponse(
                status_code=502,
                content={"detail": f"Analysis returned unparseable text: {content[:300]}"},
            )
        content = parsed

    # Qwen sometimes wraps the JSON object in an array ([{...}]) despite
    # instructions. Unwrap it so the frontend always receives an object.
    if isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict):
        content = content[0]

    if not isinstance(content, dict):
        return JSONResponse(
            status_code=502,
            content={"detail": "Analysis returned an unexpected format (expected a JSON object)."},
        )

    # The leader sometimes paraphrases a member's response instead of copying
    # the raw JSON (e.g. the feedback scores). When a field is not valid JSON,
    # fall back to the member's raw structured response.
    member_by_name = {}
    for member_response in getattr(response, "member_responses", None) or []:
        member_name = getattr(member_response, "agent_name", "") or ""
        if member_name:
            member_by_name[member_name.lower()] = member_response

    def _json_string_of(value) -> str:
        if isinstance(value, str):
            return value
        if hasattr(value, "model_dump_json"):
            return value.model_dump_json()
        try:
            return json.dumps(value)
        except TypeError:
            return str(value)

    for field_name, member_keyword in (
        ("voice_analysis_response", "voice"),
        ("feedback_response", "feedback"),
    ):
        value = content.get(field_name)
        if not isinstance(value, str) or _extract_json_value(value) is None:
            for member_name, member_response in member_by_name.items():
                if member_keyword in member_name:
                    raw = getattr(member_response, "content", None)
                    if raw is not None:
                        content[field_name] = _json_string_of(raw)
                    break

    try:
        json_compatible_response = jsonable_encoder(content)
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"detail": f"Could not serialize the analysis result: {e}"},
        )

    return JSONResponse(content=json_compatible_response)
