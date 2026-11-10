"""理赔模块的数据访问层。"""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.claim.models import Claim


class ClaimRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, **kwargs) -> Claim:
        claim = Claim(**kwargs)
        self.session.add(claim)
        await self.session.flush()
        return claim

    async def get_by_no(self, claim_no: str) -> Claim | None:
        return await self.session.scalar(select(Claim).where(Claim.claim_no == claim_no))

    async def list_by_user(
        self, user_id: int, *, status: str | None = None, page: int = 1, page_size: int = 20
    ) -> tuple[list[Claim], int]:
        conditions = [Claim.user_id == user_id]
        if status:
            conditions.append(Claim.status == status)

        total = await self.session.scalar(
            select(func.count()).select_from(Claim).where(*conditions)
        )
        stmt = (
            select(Claim)
            .where(*conditions)
            .order_by(Claim.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        items = list((await self.session.scalars(stmt)).all())
        return items, int(total or 0)
