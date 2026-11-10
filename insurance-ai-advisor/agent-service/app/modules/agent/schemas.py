"""Agent 模块的请求与响应模型。"""

from pydantic import BaseModel, Field


class ClauseHit(BaseModel):
    """一条检索结果。带上各路排名是为了让"为什么它排第一"可解释。

    调试检索效果时，只看最终顺序是看不出问题的；
    看到 dense_rank=3、sparse_rank=None，就知道这条是纯语义召回的。
    """

    chunk_id: int
    clause_id: int
    product_id: int
    clause_no: str
    title: str = ""
    product_name: str
    content: str
    score: float = Field(..., description="最终排序分数")
    rrf_score: float = Field(..., description="RRF 融合得分")
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None


class ClauseSearchIn(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="用户问题")
    top_n: int | None = Field(None, ge=1, le=20, description="返回条数，默认取配置值")
    use_rerank: bool | None = Field(None, description="是否重排，默认取配置值")
    product_id: int | None = Field(None, description="限定在某个产品内检索")
    mode: str = Field(
        "hybrid",
        pattern="^(hybrid|dense|sparse)$",
        description="检索模式：hybrid 混合 / dense 仅语义 / sparse 仅关键词。用于效果对比",
    )


class ClauseSearchOut(BaseModel):
    query: str
    mode: str
    used_rerank: bool
    hits: list[ClauseHit]
    dense_count: int = Field(..., description="稠密检索召回的候选数")
    sparse_count: int = Field(..., description="稀疏检索召回的候选数")
    merged_count: int = Field(..., description="融合后的去重候选数")


class ClauseCitation(BaseModel):
    """回答引用的条款出处。前端据此渲染"依据"折叠区。"""

    clause_id: int
    product_name: str
    clause_no: str
    title: str = ""
    content: str


class ToolCallTrace(BaseModel):
    """这一轮 Agent 调用了哪些工具、传了什么参数、拿到什么结果。"""

    name: str
    arguments: dict
    result_preview: str


class ChatIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="用户提问")
    conversation_id: int | None = Field(
        None, description="会话 id。不传就新建一个会话，传了就接在之前的对话后面"
    )


class ChatOut(BaseModel):
    conversation_id: int
    message_id: int
    question: str
    answer: str
    intent: str = Field(..., description="意图路由结果，用于调试和效果分析")
    citations: list[ClauseCitation] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    latency_ms: int
    total_tokens: int | None = None
