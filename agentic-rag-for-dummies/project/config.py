import os

# --- Directory Configuration ---
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MARKDOWN_DIR = os.path.join(_BASE_DIR, "markdown_docs")
PARENT_STORE_PATH = os.path.join(_BASE_DIR, "parent_store")
QDRANT_DB_PATH = os.path.join(_BASE_DIR, "qdrant_db")

# --- Qdrant Configuration ---
CHILD_COLLECTION = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"

# --- Model Configuration ---
# Local path (not a HF repo id): loading from a directory never contacts the
# HuggingFace Hub, so evaluation works fully offline.
DENSE_MODEL = "E:/models/Qwen3-Embedding-0.6B"
SPARSE_MODEL = "Qdrant/bm25"

# LLM: DashScope (Alibaba Cloud Qwen, OpenAI-compatible endpoint)
# Put your API key in project/.env as DASHSCOPE_API_KEY=sk-...
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
LLM_MODEL = "qwen3.8-flash"
# RAGAS judge model (OpenAI-compatible, served via DashScope in this setup).
JUDGE_MODEL = "qwen3.7-flash"
LLM_TEMPERATURE = 0

# --- Retrieval Configuration ---
RETRIEVAL_SCORE_THRESHOLD = 0.4
DEFAULT_RETRIEVAL_K = 7
CHILD_CHUNK_SEPARATOR = "\n\n<CHILD_CHUNK_BOUNDARY>\n\n"

# --- Reranker Configuration ---
# bge-reranker-base cross-encoder, loaded from a local model directory.
RERANKER_ENABLED = True
RERANKER_MODEL_PATH = "E:/models/bge-reranker-base"
RERANK_TOP_K = 5

# --- Agent Configuration ---
MAX_TOOL_CALLS = 8
MAX_ITERATIONS = 10
GRAPH_RECURSION_LIMIT = 50
MAIN_HISTORY_MESSAGES_TO_KEEP = 4
BASE_TOKEN_THRESHOLD = 2000
TOKEN_GROWTH_FACTOR = 0.9

# --- Terminal Execution Logging ---
EXECUTION_LOGGING_ENABLED = False
EXECUTION_LOG_MAX_CHARS = 1200
EXECUTION_LOG_USE_COLOR = True

# --- Text Splitter Configuration ---
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 3000
MAX_PARENT_SIZE = 6000
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]

# --- Langfuse Observability ---
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")
