"""用户模块的数据访问层。

这一层的唯一职责：读写数据库。不判断业务规则、不抛业务异常。
好处是以后换 ORM、加缓存、加读写分离，都只改这一个文件。

事务由 service 统一提交，repository 只负责 add / flush。
这样"一次请求里的多个写操作"才能是一个原子操作。
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.user.models import User


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, user_id: int) -> User | None:
        return await self.session.scalar(select(User).where(User.id == user_id))

    async def get_by_phone(self, phone: str) -> User | None:
        return await self.session.scalar(select(User).where(User.phone == phone))

    async def create(
        self,
        *,
        phone: str,
        password_hash: str,
        nickname: str | None = None,
        real_name: str | None = None,
        id_card_no: str | None = None,
    ) -> User:
        user = User(
            phone=phone,
            password_hash=password_hash,
            nickname=nickname,
            real_name=real_name,
            id_card_no=id_card_no,
        )
        self.session.add(user)
        await self.session.flush()  # 拿到自增 id 和数据库默认值，暂不提交
        return user
