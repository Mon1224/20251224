"""保单表（insurance_biz.policies）。

保单和订单的区别（保险行业的核心概念）：
  - orders  是交易行为：用户点了投保、付了钱。支付失败订单依然存在。
  - policies 是合同状态：付款完成、核保通过后才生成，是理赔的唯一依据。
所以理赔时问的是"哪张保单"，而不是"哪个订单"。
"""

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import Date, ForeignKey, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.mixins import TimestampMixin
from app.infrastructure.database import BizBase

if TYPE_CHECKING:
    from app.modules.product.models import Product


class Policy(TimestampMixin, BizBase):
    __tablename__ = "policies"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    policy_no: Mapped[str] = mapped_column(String(32), unique=True, index=True, comment="保单号")
    order_id: Mapped[int] = mapped_column(
        ForeignKey("orders.id"), unique=True, comment="一张订单对应一张保单"
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"))

    holder_name: Mapped[str] = mapped_column(String(50), comment="投保人")
    holder_id_card: Mapped[str] = mapped_column(String(32), comment="投保人身份证号")
    insured_name: Mapped[str] = mapped_column(String(50), comment="被保人")
    insured_id_card: Mapped[str] = mapped_column(String(32), comment="被保人身份证号")
    beneficiary: Mapped[str] = mapped_column(String(50), default="法定", comment="受益人")

    coverage_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), comment="保额（元）")
    premium: Mapped[Decimal] = mapped_column(Numeric(12, 2), comment="保费（元/年）")

    start_date: Mapped[date] = mapped_column(Date, comment="生效日")
    end_date: Mapped[date] = mapped_column(Date, comment="到期日")

    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        server_default="pending",
        index=True,
        comment=(
            "pending 待生效 / active 生效中 / expired 已失效 / "
            "claimed 已理赔 / surrendered 已退保"
        ),
    )

    product: Mapped["Product"] = relationship(lazy="selectin")

    @property
    def product_name(self) -> str | None:
        return self.product.name if self.product else None

    def __repr__(self) -> str:
        return f"<Policy {self.policy_no} {self.status}>"
