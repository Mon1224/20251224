"""订单模块的业务逻辑层。

这里是"传统保险业务"的核心流程：
    选产品 → 算保费 → 生成订单 → 支付 → 自动出保单
"""

import logging
from datetime import UTC, date, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.common.utils import (
    add_years,
    generate_order_no,
    generate_policy_no,
    parse_term_options,
)
from app.core.exceptions import BusinessError, NotFoundError
from app.modules.order.models import Order
from app.modules.order.repository import OrderRepository
from app.modules.order.schemas import OrderCreateIn
from app.modules.policy.models import Policy
from app.modules.policy.repository import PolicyRepository
from app.modules.product.repository import ProductRepository
from app.modules.product.service import calc_premium
from app.modules.user.models import User

logger = logging.getLogger(__name__)


class OrderService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repo = OrderRepository(session)
        self.product_repo = ProductRepository(session)
        self.policy_repo = PolicyRepository(session)

    async def create(self, user: User, payload: OrderCreateIn) -> Order:
        """下单。此时订单状态是 pending（待支付），还没有保单。"""
        product = await self.product_repo.get_by_id(payload.product_id)
        if product is None or product.status != "on_sale":
            raise NotFoundError("产品不存在或已下架")

        if not (product.age_min <= payload.insured_age <= product.age_max):
            raise BusinessError(
                f"该产品投保年龄为 {product.age_min}-{product.age_max} 岁，"
                f"被保人 {payload.insured_age} 岁不符合条件",
                code="AGE_NOT_ALLOWED",
            )

        allowed_terms = parse_term_options(product.term_options)
        if allowed_terms and payload.term_years not in allowed_terms:
            raise BusinessError(
                f"该产品可选保险期间为 {product.term_options} 年", code="TERM_NOT_ALLOWED"
            )

        premium, _, _, _ = calc_premium(
            product, payload.insured_age, payload.coverage_amount, payload.term_years
        )

        order = await self.repo.create(
            order_no=generate_order_no(),
            user_id=user.id,
            product_id=product.id,
            premium=premium,
            coverage_amount=payload.coverage_amount,
            term_years=payload.term_years,
            insured_name=payload.insured_name,
            insured_id_card=payload.insured_id_card,
            insured_age=payload.insured_age,
            status="pending",
        )
        await self.session.commit()
        await self.session.refresh(order)

        logger.info("下单成功 order_no=%s 保费=%s", order.order_no, order.premium)
        return order

    async def pay(self, user: User, order_no: str) -> tuple[Order, Policy]:
        """支付并出保单。

        **订单更新和保单创建必须一次提交**：中间任何一步失败都要整体回滚，
        否则会出现"钱付了但没有保单"这种最严重的业务事故。
        """
        order = await self._get_own_order(user.id, order_no)
        if order.status != "pending":
            raise BusinessError(
                f"订单当前是「{order.status}」状态，不能支付", code="ORDER_NOT_PAYABLE"
            )

        if not user.real_name or not user.id_card_no:
            raise BusinessError(
                "请先完善实名信息（真实姓名和身份证号）再支付", code="REAL_NAME_REQUIRED"
            )

        order.status = "paid"
        order.paid_at = datetime.now(UTC)

        start = date.today() + timedelta(days=1)  # 保险惯例：次日零时生效
        policy = await self.policy_repo.create(
            policy_no=generate_policy_no(),
            order_id=order.id,
            user_id=user.id,
            product_id=order.product_id,
            holder_name=user.real_name,
            holder_id_card=user.id_card_no,
            insured_name=order.insured_name,
            insured_id_card=order.insured_id_card,
            coverage_amount=order.coverage_amount,
            premium=order.premium,
            start_date=start,
            end_date=add_years(start, order.term_years),
            status="active",
        )

        await self.session.commit()  # 一次提交，两处修改同时生效
        await self.session.refresh(order)
        await self.session.refresh(policy)

        logger.info("支付成功 order_no=%s 生成保单 policy_no=%s", order.order_no, policy.policy_no)
        return order, policy

    async def cancel(self, user: User, order_no: str) -> Order:
        order = await self._get_own_order(user.id, order_no)
        if order.status != "pending":
            raise BusinessError(
                f"订单当前是「{order.status}」状态，不能取消", code="ORDER_NOT_CANCELLABLE"
            )
        order.status = "cancelled"
        await self.session.commit()
        await self.session.refresh(order)
        return order

    async def list_orders(
        self, user_id: int, *, status: str | None, page: int, page_size: int
    ) -> tuple[list[Order], int]:
        return await self.repo.list_by_user(
            user_id, status=status, page=page, page_size=page_size
        )

    async def get_order(self, user_id: int, order_no: str) -> Order:
        return await self._get_own_order(user_id, order_no)

    async def _get_own_order(self, user_id: int, order_no: str) -> Order:
        order = await self.repo.get_by_no(order_no)
        if order is None:
            raise NotFoundError("订单不存在")
        # 越权检查：只能操作自己的订单。少了这一句就是严重的越权漏洞。
        if order.user_id != user_id:
            raise NotFoundError("订单不存在")
        return order
