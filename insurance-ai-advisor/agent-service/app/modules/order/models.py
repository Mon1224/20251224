"""订单表（insurance_biz.orders）。

订单是"交易行为"：用户点了投保、付了钱就有一条订单。
它和保单的区别见 policy/models.py。
"""

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.mixins import TimestampMixin
from app.infrastructure.database import BizBase

if TYPE_CHECKING:  # 只在类型检查时导入，避免运行时的循环导入
    from app.modules.product.models import Product


class Order(TimestampMixin, BizBase):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    order_no: Mapped[str] = mapped_column(String(32), unique=True, index=True, comment="订单号")
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)

    premium: Mapped[Decimal] = mapped_column(Numeric(12, 2), comment="实付保费（元）")
    coverage_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), comment="投保保额（元）")
    term_years: Mapped[int] = mapped_column(Integer, comment="保险期间（年）")

    insured_name: Mapped[str] = mapped_column(String(50), comment="被保人姓名")
    insured_id_card: Mapped[str] = mapped_column(String(32), comment="被保人身份证号")
    insured_age: Mapped[int] = mapped_column(Integer, comment="被保人投保时年龄")

    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        server_default="pending",
        index=True,
        comment="pending 待支付 / paid 已支付 / cancelled 已取消",
    )
    paid_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), comment="支付时间")

    # 注意 lazy="selectin"：异步 SQLAlchemy 里默认的懒加载会抛 MissingGreenlet 错误，
    # 因为懒加载需要在同步上下文里发 SQL。这里统一改成预加载。
    product: Mapped["Product"] = relationship(lazy="selectin")

    @property
    def product_name(self) -> str | None:
        return self.product.name if self.product else None

    def __repr__(self) -> str:
        return f"<Order {self.order_no} {self.status}>"
