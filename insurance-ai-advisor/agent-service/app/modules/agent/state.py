"""Agent 的状态定义。

LangGraph 的核心概念就是"状态 + 节点 + 边"：
  状态 = 一张在节点之间传递并不断被更新的数据表
  节点 = 读状态、干活、返回增量更新的函数
  边   = 决定下一步去哪个节点

messages 用了 add_messages 归约器，意思是节点返回新消息时"追加"而不是"覆盖"，
这样多轮对话的历史才会累加。
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """图里流转的状态。total=False 表示每个字段都可以不存在。"""

    # 对话消息历史（追加式更新）
    messages: Annotated[list[AnyMessage], add_messages]

    # 本轮是谁在问
    user_id: int
    question: str

    # 意图路由结果：clause_qa / recommend / premium_calc / claim_help / chitchat
    intent: str

    # 检索到的条款上下文
    contexts: list[dict]
    # 引用的条款 id，落库到 messages.references
    references: list[int]

    # 本轮最终回答
    answer: str
