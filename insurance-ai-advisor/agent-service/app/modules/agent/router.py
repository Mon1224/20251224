"""Agent 模块的接口层：对话（流式/非流式）与条款检索接口。"""

import asyncio

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.common.deps import AgentDB, CurrentAuth
from app.infrastructure.database import AgentSessionLocal
from app.modules.agent.retriever import search_clauses_async
from app.modules.agent.schemas import ChatIn, ChatOut, ClauseSearchIn, ClauseSearchOut
from app.modules.agent.service import ChatService

router = APIRouter(prefix="/api/v1/agent", tags=["AI 顾问"])


@router.post("/chat", response_model=ChatOut, summary="与 AI 顾问对话")
async def chat(payload: ChatIn, db: AgentDB, auth: CurrentAuth) -> ChatOut:
    """完整走一遍 Agent 状态图：意图路由 → 检索或工具调用 → 生成回答。

    不传 conversation_id 就新建会话，传了就接在之前的对话后面（多轮）。

    为了能在问题发生后还原现场，本轮还会写三张表：
    messages（问答内容）、agent_runs（意图/耗时/token）、tool_calls（工具调用明细）。
    """
    return await ChatService(db).chat(auth.user.id, payload.question, payload.conversation_id)


@router.post("/chat/stream", summary="流式对话（SSE）")
async def chat_stream(payload: ChatIn, auth: CurrentAuth) -> EventSourceResponse:
    """边生成边推送给前端。

    事件类型：meta / status / delta / done / error，详见 ChatService.astream_chat。

    **注意这里没有用 AgentDB 依赖**：流式响应是在接口函数返回之后才开始推送的，
    而 FastAPI 的会话依赖会在响应结束后关闭，两者时序对不上。
    所以数据库会话必须在生成器内部自己开、自己关。
    """
    user_id = auth.user.id  # 先把要用的数据取出来，之后就不依赖请求作用域的会话了

    async def event_generator():
        session = AgentSessionLocal()
        try:
            service = ChatService(session)
            async for event in service.astream_chat(
                user_id, payload.question, payload.conversation_id
            ):
                yield event
        finally:
            # 客户端断开时当前任务已被取消，直接 await session.close() 会被中断，
            # 连接就归还不了连接池（SQLAlchemy 会打一条 "Exception terminating connection"）。
            # shield 让关闭动作在独立任务里跑完。
            try:
                await asyncio.shield(session.close())
            except BaseException:  # noqa: BLE001
                pass

    return EventSourceResponse(
        event_generator(),
        # X-Accel-Buffering 是给 nginx 看的：不加的话 nginx 会把流缓冲起来，流式就废了
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/clause-search",
    response_model=ClauseSearchOut,
    summary="条款检索（混合检索 + 重排）",
)
async def clause_search(payload: ClauseSearchIn, auth: CurrentAuth) -> ClauseSearchOut:
    """检索接口。

    mode 参数是给调试用的：
      dense  —— 只看语义检索的效果
      sparse —— 只看 BM25 关键词检索的效果
      hybrid —— 两路融合 + 重排（默认，也是线上用的）
    把三种模式跑同一批问题，就能直观看出融合到底带来了什么。
    """
    return await search_clauses_async(
        payload.query,
        mode=payload.mode,
        final_top_n=payload.top_n,
        use_rerank=payload.use_rerank,
        product_id=payload.product_id,
    )
