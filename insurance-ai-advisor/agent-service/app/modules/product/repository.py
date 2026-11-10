"""产品模块的数据访问层。"""

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.modules.product.models import Clause, Product


class ProductRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list_products(
        self,
        *,
        category: str | None = None,
        keyword: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Product], int]:
        """返回 (当页数据, 总数)。总数单独查一次，前端要用来算页数。"""
        conditions = [Product.status == "on_sale"]
        if category:
            conditions.append(Product.category == category)
        if keyword:
            like = f"%{keyword}%"
            conditions.append(
                or_(Product.name.ilike(like), Product.company.ilike(like), Product.tags.ilike(like))
            )

        total = await self.session.scalar(
            select(func.count()).select_from(Product).where(*conditions)
        )

        stmt = (
            select(Product)
            .where(*conditions)
            .order_by(Product.premium_min)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        items = list((await self.session.scalars(stmt)).all())
        return items, int(total or 0)

    async def get_by_id(self, product_id: int, *, with_clauses: bool = False) -> Product | None:
        stmt = select(Product).where(Product.id == product_id)
        if with_clauses:
            # 一次性把条款查出来，避免 N+1 查询
            stmt = stmt.options(selectinload(Product.clauses))
        return await self.session.scalar(stmt)

    async def get_by_code(self, code: str) -> Product | None:
        return await self.session.scalar(select(Product).where(Product.code == code))

    async def list_clauses(self, product_id: int) -> list[Clause]:
        stmt = select(Clause).where(Clause.product_id == product_id).order_by(Clause.clause_no)
        return list((await self.session.scalars(stmt)).all())
