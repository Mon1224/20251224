"""Agent 运行记录与工具调用记录（insurance_agent 库）。

这两张表是"给人看的运行日志"，用于排查问题、成本核算、效果分析。
注意和 checkpointer 的区别：checkpointer 存的是"给 LangGraph 用的内部状态"，
目的是让流程图能恢复执行；而这里是业务可观测性数据，两者不能互相替代。
"""

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.mixins import TimestampMixin
from app.infrastructure.database import AgentBase


class AgentRun(TimestampMixin, AgentBase):
    """一次 Agent 执行 = 一行。每调用一次 LangGraph 图就写一条。"""

    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    message_id: Mapped[int | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL"), comment="对应的助手消息"
    )

    intent: Mapped[str | None] = mapped_column(
        String(32), index=True, comment="意图路由结果：clause_qa/recommend/claim_help/chitchat"
    )
    model: Mapped[str | None] = mapped_column(String(50), comment="使用的模型名")

    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)

    latency_ms: Mapped[int | None] = mapped_column(Integer, comment="端到端耗时（毫秒）")
    retrieved_clause_ids: Mapped[list | None] = mapped_column(
        JSONB, comment="本次检索命中的条款 id，用于复盘回答依据"
    )

    status: Mapped[str] = mapped_column(
        String(20), default="success", server_default="success", comment="success / failed"
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    tool_calls: Mapped[list["ToolCall"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<AgentRun id={self.id} intent={self.intent} {self.status}>"


class ToolCall(TimestampMixin, AgentBase):
    """一次 Agent 运行中的单次工具调用明细。"""

    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True
    )
    tool_name: Mapped[str] = mapped_column(
        String(64),
        comment="工具名，见 app/modules/agent/tools.py 的 BUSINESS_TOOLS",
    )
    arguments: Mapped[dict | None] = mapped_column(JSONB, comment="调用入参")
    result: Mapped[dict | None] = mapped_column(JSONB, comment="返回结果，便于复现问题")
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    success: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    error_message: Mapped[str | None] = mapped_column(Text)

    run: Mapped["AgentRun"] = relationship(back_populates="tool_calls")

    def __repr__(self) -> str:
        return f"<ToolCall {self.tool_name} success={self.success}>"
