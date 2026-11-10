"""理赔模块的接口层。"""

from fastapi import APIRouter, Query, status

from app.common.deps import BizDB, CurrentAuth
from app.common.pagination import Page
from app.modules.claim.schemas import ClaimCreateIn, ClaimOut
from app.modules.claim.service import ClaimService

router = APIRouter(prefix="/api/v1/claims", tags=["理赔"])


@router.post(
    "",
    response_model=ClaimOut,
    status_code=status.HTTP_201_CREATED,
    summary="提交理赔申请",
)
async def submit_claim(payload: ClaimCreateIn, db: BizDB, auth: CurrentAuth) -> ClaimOut:
    claim = await ClaimService(db).submit(auth.user.id, payload)
    return ClaimOut.model_validate(claim)


@router.get("", response_model=Page[ClaimOut], summary="我的理赔记录")
async def list_claims(
    db: BizDB,
    auth: CurrentAuth,
    status_filter: str | None = Query(None, alias="status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> Page[ClaimOut]:
    items, total = await ClaimService(db).list_claims(
        auth.user.id, status=status_filter, page=page, page_size=page_size
    )
    return Page[ClaimOut](
        items=[ClaimOut.model_validate(i) for i in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{claim_no}", response_model=ClaimOut, summary="理赔详情")
async def get_claim(claim_no: str, db: BizDB, auth: CurrentAuth) -> ClaimOut:
    return ClaimOut.model_validate(await ClaimService(db).get_claim(auth.user.id, claim_no))
