"""保单模块的数据访问层。"""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.policy.models import Policy


class PolicyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, **kwargs) -> Policy:
        policy = Policy(**kwargs)
        self.session.add(policy)
        await self.session.flush()
        return policy

    async def get_by_no(self, policy_no: str) -> Policy | None:
        return await self.session.scalar(select(Policy).where(Policy.policy_no == policy_no))

    async def list_by_user(
        self, user_id: int, *, status: str | None = None, page: int = 1, page_size: int = 20
    ) -> tuple[list[Policy], int]:
        conditions = [Policy.user_id == user_id]
        if status:
            conditions.append(Policy.status == status)

        total = await self.session.scalar(
            select(func.count()).select_from(Policy).where(*conditions)
        )
        stmt = (
            select(Policy)
            .where(*conditions)
            .order_by(Policy.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        items = list((await self.session.scalars(stmt)).all())
        return items, int(total or 0)
