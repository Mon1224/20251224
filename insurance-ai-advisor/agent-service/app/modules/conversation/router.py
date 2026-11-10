"""会话模块的接口层。"""

from fastapi import APIRouter, Query

from app.common.deps import AgentDB, CurrentAuth
from app.common.pagination import Page
from app.core.exceptions import NotFoundError
from app.modules.conversation.repository import ConversationRepository, MessageRepository
from app.modules.conversation.schemas import ConversationOut, MessageOut

router = APIRouter(prefix="/api/v1/conversations", tags=["会话"])


@router.get("", response_model=Page[ConversationOut], summary="我的会话列表")
async def list_conversations(
    db: AgentDB,
    auth: CurrentAuth,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> Page[ConversationOut]:
    items, total = await ConversationRepository(db).list_by_user(
        auth.user.id, page=page, page_size=page_size
    )
    return Page[ConversationOut](
        items=[ConversationOut.model_validate(i) for i in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{conversation_id}/messages",
    response_model=Page[MessageOut],
    summary="会话的消息记录",
)
async def list_messages(
    conversation_id: int,
    db: AgentDB,
    auth: CurrentAuth,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
) -> Page[MessageOut]:
    conversation = await ConversationRepository(db).get(conversation_id)
    if conversation is None or conversation.user_id != auth.user.id:
        raise NotFoundError("会话不存在")

    items, total = await MessageRepository(db).list_by_conversation(
        conversation_id, page=page, page_size=page_size
    )
    return Page[MessageOut](
        items=[MessageOut.model_validate(i) for i in items],
        total=total,
        page=page,
        page_size=page_size,
    )
