"""订单模块的数据访问层。"""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.order.models import Order


class OrderRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, **kwargs) -> Order:
        order = Order(**kwargs)
        self.session.add(order)
        await self.session.flush()
        return order

    async def get_by_no(self, order_no: str) -> Order | None:
        return await self.session.scalar(select(Order).where(Order.order_no == order_no))

    async def list_by_user(
        self, user_id: int, *, status: str | None = None, page: int = 1, page_size: int = 20
    ) -> tuple[list[Order], int]:
        conditions = [Order.user_id == user_id]
        if status:
            conditions.append(Order.status == status)

        total = await self.session.scalar(
            select(func.count()).select_from(Order).where(*conditions)
        )
        stmt = (
            select(Order)
            .where(*conditions)
            .order_by(Order.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        items = list((await self.session.scalars(stmt)).all())
        return items, int(total or 0)
