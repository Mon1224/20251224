"""会话与消息的数据访问层。"""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.conversation.models import Conversation, Message


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, conversation_id: int) -> Conversation | None:
        return await self.session.scalar(
            select(Conversation).where(Conversation.id == conversation_id)
        )

    async def list_by_user(
        self, user_id: int, *, page: int = 1, page_size: int = 20
    ) -> tuple[list[Conversation], int]:
        from sqlalchemy import func

        total = await self.session.scalar(
            select(func.count()).select_from(Conversation).where(Conversation.user_id == user_id)
        )
        stmt = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        return list((await self.session.scalars(stmt)).all()), int(total or 0)

    async def create(self, user_id: int, title: str) -> Conversation:
        conversation = Conversation(
            user_id=user_id,
            title=title[:200] or "新的对话",
            message_count=0,
            last_message_at=datetime.now(UTC),
            status="active",
        )
        self.session.add(conversation)
        await self.session.flush()
        return conversation

    async def touch(self, conversation: Conversation, added: int = 2) -> None:
        """更新会话的最后活跃时间和消息数。"""
        conversation.message_count += added
        conversation.last_message_at = datetime.now(UTC)
        await self.session.flush()


class MessageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(
        self,
        conversation_id: int,
        role: str,
        content: str,
        *,
        token_count: int | None = None,
        references: list[int] | None = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            token_count=token_count,
            references=references,
        )
        self.session.add(message)
        await self.session.flush()
        return message

    async def recent(self, conversation_id: int, limit: int = 10) -> list[Message]:
        """取最近若干轮消息，用于拼多轮对话上下文。

        按 id 倒序取 limit 条再反转，保证拿到的是"最新的 limit 条"且顺序正确。
        """
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id.desc())
            .limit(limit)
        )
        rows = list((await self.session.scalars(stmt)).all())
        return list(reversed(rows))

    async def list_by_conversation(
        self, conversation_id: int, *, page: int = 1, page_size: int = 50
    ) -> tuple[list[Message], int]:
        from sqlalchemy import func

        total = await self.session.scalar(
            select(func.count())
            .select_from(Message)
            .where(Message.conversation_id == conversation_id)
        )
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        return list((await self.session.scalars(stmt)).all()), int(total or 0)
