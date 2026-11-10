"""Agent 编排服务：调用状态图，并把这一轮发生的一切落库。

一次问答要写四张表：
    conversations   会话（不存在就新建，存在就更新活跃时间）
    messages        用户消息 + 助手回答
    agent_runs      本轮运行的意图、耗时、token 消耗、检索到的条款
    tool_calls      本轮每一次工具调用的入参和结果

落库不只是为了给前端看历史。agent_runs 和 tool_calls 是排查线上问题的唯一依据：
用户投诉"AI 回答错了"时，你要能还原出当时走了哪条分支、检索到哪些条款、调了什么工具。
"""

import asyncio
import json
import logging
import time

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import BusinessError, NotFoundError
from app.modules.agent.graph import get_graph
from app.modules.agent.repository import AgentRunRepository
from app.modules.agent.schemas import ChatOut, ClauseCitation, ToolCallTrace
from app.modules.conversation.repository import ConversationRepository, MessageRepository

logger = logging.getLogger(__name__)

# 带进模型的历史消息条数（一问一答算两条）
HISTORY_LIMIT = 10
RECURSION_LIMIT = 12

# 只有这两个节点的 token 才是要展示给用户的回答。
# route 节点也会调模型，但它输出的是分类标签，绝不能漏进对话气泡。
STREAM_NODES = {"generate", "agent"}

INTENT_STATUS = {
    "clause_qa": "正在检索相关保险条款…",
    "recommend": "正在查询产品库…",
    "premium_calc": "正在准备保费试算…",
    "claim_help": "正在查询您的保单与理赔记录…",
    "chitchat": "正在组织回答…",
}

TOOL_LABELS = {
    "list_insurance_products": "产品查询",
    "calculate_premium": "保费试算",
    "get_my_policies": "保单查询",
    "get_my_claims": "理赔查询",
    "get_my_orders": "订单查询",
    "search_insurance_clauses": "条款检索",
}


def _sse(event: str, data: dict) -> dict:
    """构造一条 SSE 事件。ensure_ascii=False 让中文按原样输出，体积也更小。"""
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


# 后台任务需要有人持有引用，否则可能被垃圾回收提前干掉
_background_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    """把协程丢到一个独立任务里跑，不让它跟着当前请求一起被取消。"""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _save_detached(
    conversation_id: int,
    *,
    answer: str,
    intent: str,
    contexts: list[dict],
    latency_ms: int,
    status: str,
    error_message: str | None = None,
) -> None:
    """断线或出错时的兜底落库。

    必须自己开一个新会话：调用方那个会话已经处于被取消的上下文里，不能再用。
    整段再套一层 try/except——"保存失败"绝不该盖住原始错误。
    """
    from app.infrastructure.database import AgentSessionLocal

    try:
        async with AgentSessionLocal() as session:
            service = ChatService(session)
            fresh = await service.conv_repo.get(conversation_id)
            if fresh is None:
                return
            await service._save_turn(
                fresh,
                answer=answer,
                intent=intent,
                contexts=contexts,
                tool_calls=[],
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=latency_ms,
                status=status,
                error_message=error_message,
            )
    except Exception:  # noqa: BLE001
        logger.exception("兜底落库失败（会话 %s）", conversation_id)


def _chunk_text(chunk) -> str:
    """从流式消息块里取出文本。

    内容可能是字符串，也可能是 [{"type": "text", "text": "..."}] 这种结构化列表
    （模型同时返回文字和工具调用时会这样），两种都要能处理。
    """
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


class ChatService:
    def __init__(self, agent_db: AsyncSession) -> None:
        self.db = agent_db
        self.conv_repo = ConversationRepository(agent_db)
        self.msg_repo = MessageRepository(agent_db)
        self.run_repo = AgentRunRepository(agent_db)

    async def chat(
        self, user_id: int, question: str, conversation_id: int | None = None
    ) -> ChatOut:
        started = time.perf_counter()

        conversation = await self._resolve_conversation(user_id, conversation_id, question)

        # 先取历史再写入本轮提问，否则会把本轮问题当成历史重复一遍
        history = await self.msg_repo.recent(conversation.id, limit=HISTORY_LIMIT)
        await self.msg_repo.add(conversation.id, "user", question)
        # 提前提交：就算后面 AI 调用失败，用户的问题也已经留档
        await self.db.commit()

        try:
            result = await self._run_graph(user_id, question, history)
        except Exception as exc:  # noqa: BLE001
            latency = int((time.perf_counter() - started) * 1000)
            await self.run_repo.create_run(
                conversation_id=conversation.id,
                intent=None,
                model=settings.deepseek_model,
                latency_ms=latency,
                status="failed",
                error_message=str(exc)[:1000],
            )
            await self.db.commit()
            logger.exception("Agent 执行失败")
            raise BusinessError("AI 服务暂时不可用，请稍后重试", code="AGENT_FAILED") from exc

        return await self._persist(conversation, question, result, started)

    async def _resolve_conversation(self, user_id: int, conversation_id: int | None, question: str):
        if conversation_id is None:
            # 用首轮提问的前 30 个字做会话标题，前端列表页直接可用
            return await self.conv_repo.create(user_id, title=question[:30])

        conversation = await self.conv_repo.get(conversation_id)
        # 越权检查：不能往别人的会话里插话
        if conversation is None or conversation.user_id != user_id:
            raise NotFoundError("会话不存在")
        if conversation.status != "active":
            raise BusinessError("该会话已结束，请开启新会话", code="CONVERSATION_CLOSED")
        return conversation

    async def _run_graph(self, user_id: int, question: str, history) -> dict:
        messages: list[BaseMessage] = []
        for record in history:
            if record.role == "user":
                messages.append(HumanMessage(content=record.content))
            elif record.role == "assistant":
                messages.append(AIMessage(content=record.content))
        messages.append(HumanMessage(content=question))

        return await get_graph().ainvoke(
            {"messages": messages, "user_id": user_id, "question": question},
            config={"recursion_limit": RECURSION_LIMIT},
        )

    async def _persist(self, conversation, question: str, result: dict, started: float) -> ChatOut:
        latency_ms = int((time.perf_counter() - started) * 1000)
        answer = result.get("answer") or self._last_ai_text(result.get("messages", []))
        intent = result.get("intent", "")
        contexts = result.get("contexts") or []
        citations = [
            ClauseCitation(
                clause_id=c["clause_id"],
                product_name=c["product_name"],
                clause_no=c["clause_no"],
                title=c.get("title", ""),
                content=c["content"],
            )
            for c in contexts
        ]
        tool_calls = self._extract_tool_calls(result.get("messages", []))
        prompt_tokens, completion_tokens = self._sum_tokens(result.get("messages", []))

        message_id = await self._save_turn(
            conversation,
            answer=answer,
            intent=intent,
            contexts=contexts,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

        logger.info(
            "问答完成 conversation=%s intent=%s 耗时=%sms tokens=%s 工具=%s",
            conversation.id,
            intent,
            latency_ms,
            prompt_tokens + completion_tokens,
            [c.name for c in tool_calls],
        )

        return ChatOut(
            conversation_id=conversation.id,
            message_id=message_id,
            question=question,
            answer=answer,
            intent=intent,
            citations=citations,
            tool_calls=tool_calls,
            latency_ms=latency_ms,
            total_tokens=(prompt_tokens + completion_tokens) or None,
        )

    async def _save_turn(
        self,
        conversation,
        *,
        answer: str,
        intent: str,
        contexts: list[dict],
        tool_calls: list[ToolCallTrace],
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        status: str = "success",
        error_message: str | None = None,
    ) -> int:
        """把这一轮问答落库，返回助手消息 id。流式和非流式两条路径共用。"""
        clause_ids = [c["clause_id"] for c in contexts] or None

        assistant_msg = await self.msg_repo.add(
            conversation.id,
            "assistant",
            answer or "(空回答)",
            token_count=completion_tokens or None,
            references=clause_ids,
        )
        await self.conv_repo.touch(conversation, added=2)

        run = await self.run_repo.create_run(
            conversation_id=conversation.id,
            message_id=assistant_msg.id,
            intent=intent or None,
            model=settings.deepseek_model,
            prompt_tokens=prompt_tokens or None,
            completion_tokens=completion_tokens or None,
            total_tokens=(prompt_tokens + completion_tokens) or None,
            latency_ms=latency_ms,
            retrieved_clause_ids=clause_ids,
            status=status,
            error_message=error_message,
        )
        for call in tool_calls:
            await self.run_repo.add_tool_call(
                run.id,
                tool_name=call.name,
                arguments=call.arguments,
                result={"preview": call.result_preview},
                success=not call.result_preview.startswith('{"error"'),
            )

        await self.db.commit()
        await self.db.refresh(assistant_msg)
        return assistant_msg.id

    # ------------------------------------------------------------ 流式输出
    async def astream_chat(self, user_id: int, question: str, conversation_id: int | None = None):
        """流式对话，逐条吐出 SSE 事件。

        事件协议（前端按 event 字段分发）：
          meta   会话 id，客户端要记住它才能多轮对话
          status 阶段提示：路由中 / 检索中 / 调用工具中 / 生成中
          delta  回答的一个片段，前端追加到气泡里
          done   结束，带上引用条款、工具轨迹、耗时、token
          error  出错
        """
        started = time.perf_counter()
        conversation = await self._resolve_conversation(user_id, conversation_id, question)
        conversation_id = conversation.id  # 断线时 ORM 对象可能已不可用，先取出主键
        history = await self.msg_repo.recent(conversation.id, limit=HISTORY_LIMIT)
        await self.msg_repo.add(conversation.id, "user", question)
        # 先提交：客户端可能中途断开，但用户的问题必须留档
        await self.db.commit()

        yield _sse("meta", {"conversation_id": conversation.id})

        messages: list[BaseMessage] = []
        for record in history:
            if record.role == "user":
                messages.append(HumanMessage(content=record.content))
            elif record.role == "assistant":
                messages.append(AIMessage(content=record.content))
        messages.append(HumanMessage(content=question))

        collected: dict = {}
        streamed: list[str] = []
        prompt_tokens = completion_tokens = 0
        announced_generating = False
        # 记录工具调用：模型请求调用是一条带 tool_calls 的 AIMessage，
        # 执行结果是一条带 tool_call_id 的 ToolMessage，两者靠 id 配对。
        planned: dict[str, tuple[str, dict]] = {}
        tool_results: dict[str, str] = {}
        tool_order: list[str] = []

        try:
            async for mode, chunk in get_graph().astream(
                {"messages": messages, "user_id": user_id, "question": question},
                stream_mode=["updates", "messages"],
                config={"recursion_limit": RECURSION_LIMIT},
            ):
                if mode == "messages":
                    message_chunk, metadata = chunk
                    # 只要最终回答节点的 token，路由节点的分类输出不能漏给用户
                    if metadata.get("langgraph_node") not in STREAM_NODES:
                        continue

                    usage = getattr(message_chunk, "usage_metadata", None)
                    if usage:
                        prompt_tokens += int(usage.get("input_tokens", 0) or 0)
                        completion_tokens += int(usage.get("output_tokens", 0) or 0)

                    text = _chunk_text(message_chunk)
                    if not text:
                        continue
                    if not announced_generating:
                        announced_generating = True
                        yield _sse("status", {"stage": "generating", "text": "正在生成回答…"})
                    streamed.append(text)
                    yield _sse("delta", {"text": text})
                    continue

                # mode == "updates"：节点产出的是"增量状态"，据此推进阶段提示
                for node, update in (chunk or {}).items():
                    if not isinstance(update, dict):
                        continue
                    if node == "route":
                        intent = update.get("intent", "")
                        collected["intent"] = intent
                        yield _sse(
                            "status",
                            {"stage": "routing", "text": INTENT_STATUS.get(intent, "正在理解…")},
                        )
                    elif node == "retrieve":
                        collected["contexts"] = update.get("contexts") or []
                    elif node == "agent":
                        names = []
                        for message in update.get("messages", []):
                            for call in getattr(message, "tool_calls", None) or []:
                                call_id = call.get("id") or call["name"]
                                planned[call_id] = (call["name"], call.get("args") or {})
                                tool_order.append(call_id)
                                names.append(TOOL_LABELS.get(call["name"], call["name"]))
                        if names:
                            yield _sse(
                                "status",
                                {"stage": "tool", "text": f"正在调用：{'、'.join(names)}…"},
                            )
                    elif node == "tools":
                        for message in update.get("messages", []):
                            if isinstance(message, ToolMessage):
                                tool_results[message.tool_call_id] = str(message.content)
                    if "answer" in update:
                        collected["answer"] = update["answer"]

        except asyncio.CancelledError:
            # 客户端断开。此刻当前任务已被取消，任何 await 都会立刻再抛 CancelledError，
            # 所以必须把落库交给独立任务，否则连数据库连接都建立不起来。
            _spawn(
                _save_detached(
                    conversation_id,
                    answer=collected.get("answer") or "".join(streamed),
                    intent=collected.get("intent", ""),
                    contexts=collected.get("contexts") or [],
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    status="partial",
                )
            )
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("流式对话失败")
            await _save_detached(
                conversation_id,
                answer=collected.get("answer") or "".join(streamed),
                intent=collected.get("intent", ""),
                contexts=collected.get("contexts") or [],
                latency_ms=int((time.perf_counter() - started) * 1000),
                status="failed",
                error_message=str(exc),
            )
            yield _sse(
                "error",
                {"code": "AGENT_FAILED", "message": "AI 服务暂时不可用，请稍后重试"},
            )
            return

        latency_ms = int((time.perf_counter() - started) * 1000)
        answer = collected.get("answer") or "".join(streamed)
        contexts = collected.get("contexts") or []
        citations = [
            ClauseCitation(
                clause_id=c["clause_id"],
                product_name=c["product_name"],
                clause_no=c["clause_no"],
                title=c.get("title", ""),
                content=c["content"],
            )
            for c in contexts
        ]
        tool_calls = [
            ToolCallTrace(
                name=planned[cid][0],
                arguments=planned[cid][1],
                result_preview=tool_results.get(cid, "")[:600],
            )
            for cid in tool_order
        ]

        try:
            message_id = await self._save_turn(
                conversation,
                answer=answer,
                intent=collected.get("intent", ""),
                contexts=contexts,
                tool_calls=tool_calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )
        except Exception:  # noqa: BLE001
            # 内容已经推给用户了，此时落库失败不应该让前端看到错误
            logger.exception("流式回答落库失败")
            yield _sse("error", {"code": "PERSIST_FAILED", "message": "回答已生成但保存失败"})
            return

        yield _sse(
            "done",
            {
                "conversation_id": conversation.id,
                "message_id": message_id,
                "intent": collected.get("intent", ""),
                "citations": [c.model_dump() for c in citations],
                "tool_calls": [t.model_dump() for t in tool_calls],
                "latency_ms": latency_ms,
                "total_tokens": (prompt_tokens + completion_tokens) or None,
            },
        )

    @staticmethod
    def _last_ai_text(messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content:
                return str(message.content)
        return ""

    @staticmethod
    def _extract_tool_calls(messages: list[BaseMessage]) -> list[ToolCallTrace]:
        """从消息流里还原出工具调用轨迹。

        模型请求调用工具是一条 AIMessage（带 tool_calls），
        工具的执行结果是一条 ToolMessage（带 tool_call_id），两者靠 id 配对。
        """
        planned: dict[str, tuple[str, dict]] = {}
        order: list[str] = []
        results: dict[str, str] = {}

        for message in messages:
            for call in getattr(message, "tool_calls", None) or []:
                call_id = call.get("id") or call["name"]
                planned[call_id] = (call["name"], call.get("args") or {})
                order.append(call_id)
            if isinstance(message, ToolMessage):
                results[message.tool_call_id] = str(message.content)

        return [
            ToolCallTrace(
                name=planned[cid][0],
                arguments=planned[cid][1],
                result_preview=results.get(cid, "")[:600],
            )
            for cid in order
            if cid in planned
        ]

    @staticmethod
    def _sum_tokens(messages: list[BaseMessage]) -> tuple[int, int]:
        """累计本轮所有模型调用的 token。

        注意是"本轮所有调用"——工具型分支里模型可能要跑好几轮
        （先决定调工具、拿到结果后再生成回答），每一轮都计费。
        """
        prompt = completion = 0
        for message in messages:
            usage = getattr(message, "usage_metadata", None)
            if usage:
                prompt += int(usage.get("input_tokens", 0) or 0)
                completion += int(usage.get("output_tokens", 0) or 0)
        return prompt, completion
