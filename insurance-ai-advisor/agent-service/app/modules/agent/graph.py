"""LangGraph 状态图：把意图路由、条款检索、业务工具组装成一张可执行的流程图。

图的形状：

    START → route ─┬─ clause_qa  → retrieve → generate → END
                   ├─ chitchat   → generate → END
                   ├─ recommend  ┐
                   ├─ premium_calc├→ agent ⇄ tools → END
                   └─ claim_help ┘

route 是岔路口，根据用户意图把人引到对应窗口（这就是"意图路由"）。
agent 节点绑定了工具，模型可以自己决定调哪个工具、传什么参数；
调完工具回到 agent 继续判断，直到它不再需要工具为止——这是标准的 ReAct 循环。
"""

import logging

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.infrastructure.llm import get_chat_model
from app.modules.agent.prompts import (
    AGENT_PROMPT,
    CHITCHAT_PROMPT,
    CLAUSE_QA_PROMPT,
    ROUTER_PROMPT,
)
from app.modules.agent.retriever import search_clauses_async
from app.modules.agent.state import AgentState
from app.modules.agent.tools import BUSINESS_TOOLS

logger = logging.getLogger(__name__)

# 允许的意图，顺序即匹配优先级
INTENTS = ("clause_qa", "premium_calc", "claim_help", "recommend", "chitchat")
DEFAULT_INTENT = "clause_qa"

# 需要调用工具的意图
TOOL_INTENTS = {"recommend", "premium_calc", "claim_help"}


async def route_node(state: AgentState) -> dict:
    """意图路由：判断用户这句话该走哪条分支。"""
    question = state["question"]
    llm = get_chat_model(temperature=0, max_tokens=16)

    try:
        raw = (await llm.ainvoke(ROUTER_PROMPT.format(question=question))).content
        text = str(raw).strip().lower()
    except Exception as exc:  # noqa: BLE001
        # 路由失败不能整轮崩掉，退回条款问答（最常用的分支）
        logger.warning("意图路由失败，使用默认意图：%s", exc)
        return {"intent": DEFAULT_INTENT}

    intent = DEFAULT_INTENT
    for candidate in INTENTS:
        if candidate in text:
            intent = candidate
            break

    logger.info("意图路由：%r -> %s", question[:30], intent)
    return {"intent": intent}


async def retrieve_node(state: AgentState) -> dict:
    """条款检索：混合检索 + 重排，取最相关的几条作为回答依据。"""
    result = await search_clauses_async(state["question"], mode="hybrid", final_top_n=5)
    contexts = [
        {
            "clause_id": hit.clause_id,
            "product_name": hit.product_name,
            "clause_no": hit.clause_no,
            "title": hit.title,
            "content": hit.content,
        }
        for hit in result.hits
    ]
    top_score = result.hits[0].score if result.hits else 0.0
    logger.info("检索到 %s 条条款，最高分 %.3f", len(contexts), top_score)
    return {"contexts": contexts, "references": [c["clause_id"] for c in contexts]}


async def generate_node(state: AgentState) -> dict:
    """生成回答。

    有条款上下文时走"开卷考试"模式（严格约束只能依据资料）；
    没有上下文时（闲聊分支）走轻松的招呼模式。
    """
    contexts = state.get("contexts") or []

    if contexts:
        context_text = "\n\n".join(
            f"【{c['product_name']} 第 {c['clause_no']} 条 {c.get('title') or ''}】\n{c['content']}"
            for c in contexts
        )
        system_prompt = CLAUSE_QA_PROMPT.format(context=context_text)
    else:
        system_prompt = CHITCHAT_PROMPT

    llm = get_chat_model(temperature=0.3, max_tokens=800)
    messages = [SystemMessage(content=system_prompt), *state["messages"]]
    reply = await llm.ainvoke(messages)

    return {"messages": [reply], "answer": str(reply.content)}


async def agent_node(state: AgentState) -> dict:
    """带工具的顾问节点。模型自己决定调哪个工具。"""
    llm = get_chat_model(temperature=0.3, max_tokens=800)
    bound = llm.bind_tools(BUSINESS_TOOLS)
    system_prompt = AGENT_PROMPT.format(user_id=state.get("user_id", 0))
    messages = [SystemMessage(content=system_prompt), *state["messages"]]

    reply = await bound.ainvoke(messages)

    update: dict = {"messages": [reply]}
    # 不再要求调用工具 = 这一轮想清楚了，可以出最终答案
    if not getattr(reply, "tool_calls", None):
        update["answer"] = str(reply.content)
    else:
        logger.info("模型请求调用工具：%s", [c["name"] for c in reply.tool_calls])
    return update


def route_after_route(state: AgentState) -> str:
    """路由节点之后往哪走。"""
    intent = state.get("intent", DEFAULT_INTENT)
    if intent == "clause_qa":
        return "retrieve"
    if intent in TOOL_INTENTS:
        return "agent"
    return "generate"


def should_continue(state: AgentState) -> str:
    """agent 节点之后：还要调工具，还是可以收尾了。"""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return END


def build_graph():
    """编译状态图。进程内只编译一次，编译结果可以并发复用。"""
    graph = StateGraph(AgentState)

    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(BUSINESS_TOOLS))

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        route_after_route,
        {"retrieve": "retrieve", "generate": "generate", "agent": "agent"},
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


_compiled = None


def get_graph():
    """带缓存的图实例。"""
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled
