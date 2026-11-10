"""理赔模块的业务逻辑层。

理赔成立的三个前提，缺一不可：
  1. 保单存在        —— 没有保单就没有理赔依据
  2. 保单属于本人    —— 不能拿别人的保单来理赔
  3. 保单在有效状态  —— 已失效、已退保的保单不能理赔
"""

import logging
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.common.utils import generate_claim_no
from app.core.exceptions import BusinessError, NotFoundError
from app.modules.claim.models import Claim
from app.modules.claim.repository import ClaimRepository
from app.modules.claim.schemas import ClaimCreateIn
from app.modules.policy.repository import PolicyRepository

logger = logging.getLogger(__name__)


class ClaimService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repo = ClaimRepository(session)
        self.policy_repo = PolicyRepository(session)

    async def submit(self, user_id: int, payload: ClaimCreateIn) -> Claim:
        policy = await self.policy_repo.get_by_no(payload.policy_no)
        if policy is None or policy.user_id != user_id:
            raise NotFoundError("保单不存在")

        if policy.status != "active":
            raise BusinessError(
                f"保单当前状态为「{policy.status}」，只有生效中的保单才能申请理赔",
                code="POLICY_NOT_ACTIVE",
            )

        claim = await self.repo.create(
            claim_no=generate_claim_no(),
            policy_id=policy.id,
            user_id=user_id,
            claim_type=payload.claim_type,
            amount=payload.amount,
            description=payload.description,
            status="submitted",
            submitted_at=datetime.now(UTC),
        )
        await self.session.commit()
        await self.session.refresh(claim)

        logger.info("理赔申请提交 claim_no=%s 保单=%s", claim.claim_no, policy.policy_no)
        return claim

    async def list_claims(
        self, user_id: int, *, status: str | None, page: int, page_size: int
    ) -> tuple[list[Claim], int]:
        return await self.repo.list_by_user(
            user_id, status=status, page=page, page_size=page_size
        )

    async def get_claim(self, user_id: int, claim_no: str) -> Claim:
        claim = await self.repo.get_by_no(claim_no)
        if claim is None or claim.user_id != user_id:
            raise NotFoundError("理赔记录不存在")
        return claim
