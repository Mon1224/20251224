"""会话与消息表（insurance_agent 库）。

两者是一对多：一个 conversation 是一个"聊天窗口"，里面装着多条 message。
分开存的原因：前端左侧列历史对话只读 conversations 就够了，
不用扫描几万条消息去分组。
"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.mixins import TimestampMixin
from app.infrastructure.database import AgentBase


class Conversation(TimestampMixin, AgentBase):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # 注意：user_id 指向的是另一个数据库里的用户，跨库无法建外键，只能加索引
    user_id: Mapped[int] = mapped_column(
        Integer, index=True, comment="业务库 users.id，跨库不加外键"
    )
    title: Mapped[str] = mapped_column(
        String(200), default="新的对话", comment="会话标题，一般取首轮提问的前几个字"
    )
    message_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20), default="active", server_default="active")

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Conversation id={self.id} {self.title}>"


class Message(TimestampMixin, AgentBase):
    """一条消息。role 决定它由谁产生。"""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(
        String(20), comment="user 用户 / assistant 助手 / system 系统 / tool 工具返回"
    )
    content: Mapped[str] = mapped_column(Text, comment="消息正文")
    token_count: Mapped[int | None] = mapped_column(Integer, comment="token 用量，用于成本核算")
    references: Mapped[list | None] = mapped_column(
        JSONB, comment="本条回答引用的条款 id 列表，前端展示出处用"
    )

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message conv={self.conversation_id} {self.role}>"
