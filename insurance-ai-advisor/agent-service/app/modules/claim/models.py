"""理赔记录表（insurance_biz.claims）。

AI 顾问的"理赔咨询"能力就是查这张表：用户问"我的理赔到哪一步了"，
Agent 通过工具调用读这里的数据。
"""

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.mixins import TimestampMixin
from app.infrastructure.database import BizBase

if TYPE_CHECKING:
    from app.modules.policy.models import Policy


class Claim(TimestampMixin, BizBase):
    __tablename__ = "claims"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    claim_no: Mapped[str] = mapped_column(String(32), unique=True, index=True, comment="理赔单号")
    policy_id: Mapped[int] = mapped_column(
        ForeignKey("policies.id"), index=True, comment="理赔依据：保单号"
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)

    claim_type: Mapped[str] = mapped_column(String(32), comment="出险类型：门诊/住院/重疾/意外")
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), comment="申请理赔金额（元）")
    approved_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 2), comment="实际赔付金额，审核通过后回填"
    )
    description: Mapped[str | None] = mapped_column(Text, comment="出险描述")

    status: Mapped[str] = mapped_column(
        String(20),
        default="submitted",
        server_default="submitted",
        index=True,
        comment=(
            "submitted 已提交 / reviewing 审核中 / approved 已通过 / "
            "rejected 已拒赔 / paid 已赔付"
        ),
    )
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    policy: Mapped["Policy"] = relationship(lazy="selectin")

    @property
    def policy_no(self) -> str | None:
        return self.policy.policy_no if self.policy else None

    def __repr__(self) -> str:
        return f"<Claim {self.claim_no} {self.status}>"
