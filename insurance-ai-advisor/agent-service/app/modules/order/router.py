"""订单模块的接口层。"""

from fastapi import APIRouter, Query, status

from app.common.deps import BizDB, CurrentAuth
from app.common.pagination import Page
from app.modules.order.schemas import OrderCreateIn, OrderOut, OrderPaidOut
from app.modules.order.service import OrderService

router = APIRouter(prefix="/api/v1/orders", tags=["投保订单"])


@router.post("", response_model=OrderOut, status_code=status.HTTP_201_CREATED, summary="投保下单")
async def create_order(payload: OrderCreateIn, db: BizDB, auth: CurrentAuth) -> OrderOut:
    order = await OrderService(db).create(auth.user, payload)
    return OrderOut.model_validate(order)


@router.get("", response_model=Page[OrderOut], summary="我的订单")
async def list_orders(
    db: BizDB,
    auth: CurrentAuth,
    status_filter: str | None = Query(None, alias="status", description="pending/paid/cancelled"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> Page[OrderOut]:
    items, total = await OrderService(db).list_orders(
        auth.user.id, status=status_filter, page=page, page_size=page_size
    )
    return Page[OrderOut](
        items=[OrderOut.model_validate(i) for i in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{order_no}", response_model=OrderOut, summary="订单详情")
async def get_order(order_no: str, db: BizDB, auth: CurrentAuth) -> OrderOut:
    order = await OrderService(db).get_order(auth.user.id, order_no)
    return OrderOut.model_validate(order)


@router.post("/{order_no}/pay", response_model=OrderPaidOut, summary="支付并出保单")
async def pay_order(order_no: str, db: BizDB, auth: CurrentAuth) -> OrderPaidOut:
    order, policy = await OrderService(db).pay(auth.user, order_no)
    return OrderPaidOut(
        order=OrderOut.model_validate(order),
        policy_no=policy.policy_no,
        policy_status=policy.status,
    )


@router.post("/{order_no}/cancel", response_model=OrderOut, summary="取消订单")
async def cancel_order(order_no: str, db: BizDB, auth: CurrentAuth) -> OrderOut:
    order = await OrderService(db).cancel(auth.user, order_no)
    return OrderOut.model_validate(order)
