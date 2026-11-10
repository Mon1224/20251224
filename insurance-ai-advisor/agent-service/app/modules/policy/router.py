"""保单模块的接口层。"""

from fastapi import APIRouter, Query

from app.common.deps import BizDB, CurrentAuth
from app.common.pagination import Page
from app.modules.policy.schemas import PolicyOut
from app.modules.policy.service import PolicyService

router = APIRouter(prefix="/api/v1/policies", tags=["保单"])


@router.get("", response_model=Page[PolicyOut], summary="我的保单")
async def list_policies(
    db: BizDB,
    auth: CurrentAuth,
    status_filter: str | None = Query(None, alias="status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> Page[PolicyOut]:
    items, total = await PolicyService(db).list_policies(
        auth.user.id, status=status_filter, page=page, page_size=page_size
    )
    return Page[PolicyOut](
        items=[PolicyOut.model_validate(i) for i in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{policy_no}", response_model=PolicyOut, summary="保单详情")
async def get_policy(policy_no: str, db: BizDB, auth: CurrentAuth) -> PolicyOut:
    return PolicyOut.model_validate(await PolicyService(db).get_policy(auth.user.id, policy_no))
