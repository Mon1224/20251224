import uuid
from langchain_openai import ChatOpenAI
import config
from db.vector_db_manager import VectorDbManager
from db.parent_store_manager import ParentStoreManager
from document_chunker import DocumentChunker
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph
from core.observability import Observability

class RAGSystem:

    def __init__(self, collection_name=config.CHILD_COLLECTION):
        self.collection_name = collection_name
        self.vector_db = VectorDbManager()
        self.parent_store = ParentStoreManager()
        self.chunker = DocumentChunker()
        self.observability = Observability()
        self.agent_graph = None
        self.thread_id = str(uuid.uuid4())
        self.recursion_limit = config.GRAPH_RECURSION_LIMIT

    def initialize(self):
        self.vector_db.create_collection(self.collection_name)
        collection = self.vector_db.get_collection(self.collection_name)

        if not config.DASHSCOPE_API_KEY:
            raise ValueError(
                "DASHSCOPE_API_KEY is not set. Copy project/.env.example to "
                "project/.env and set DASHSCOPE_API_KEY=sk-..."
            )

        llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,
            timeout=180,
            max_retries=1,
        )
        tools = ToolFactory(collection).create_tools()
        self.agent_graph = create_agent_graph(llm, tools)

    def get_config(self):
        cfg = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": self.recursion_limit}
        handler = self.observability.get_handler()
        if handler:
            cfg["callbacks"] = [handler]
        return cfg

    def reset_thread(self):
        try:
            self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as e:
            print(f"Warning: Could not delete thread {self.thread_id}: {e}")
        self.thread_id = str(uuid.uuid4())
