"""保单模块的业务逻辑层。

保单由 OrderService.pay() 在支付成功时自动生成，
所以这里只提供查询能力，没有"创建保单"的接口——保单不能让用户直接创建。
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.modules.policy.models import Policy
from app.modules.policy.repository import PolicyRepository


class PolicyService:
    def __init__(self, session: AsyncSession) -> None:
        self.repo = PolicyRepository(session)

    async def list_policies(
        self, user_id: int, *, status: str | None, page: int, page_size: int
    ) -> tuple[list[Policy], int]:
        return await self.repo.list_by_user(
            user_id, status=status, page=page, page_size=page_size
        )

    async def get_policy(self, user_id: int, policy_no: str) -> Policy:
        policy = await self.repo.get_by_no(policy_no)
        # 用户只能看自己的保单，别人的一律当作"不存在"
        if policy is None or policy.user_id != user_id:
            raise NotFoundError("保单不存在")
        return policy
