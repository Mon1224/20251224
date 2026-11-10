"""订单模块的请求与响应模型。"""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, computed_field

ORDER_STATUS_TEXT = {
    "pending": "待支付",
    "paid": "已支付",
    "cancelled": "已取消",
}


class OrderCreateIn(BaseModel):
    """投保请求。

    注意这里**没有**保费字段：保费由后端按产品费率算出来，
    绝不能让前端传价格，否则用户可以自己改成 1 元投保。
    """

    product_id: int
    coverage_amount: Decimal = Field(..., gt=0, description="投保保额（元）")
    term_years: int = Field(..., gt=0, description="保险期间（年）")
    insured_name: str = Field(..., min_length=2, max_length=50, description="被保人姓名")
    insured_id_card: str = Field(..., min_length=15, max_length=18, description="被保人身份证号")
    insured_age: int = Field(..., ge=0, le=100, description="被保人年龄")


class OrderOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    order_no: str
    product_id: int
    product_name: str | None = None
    premium: Decimal
    coverage_amount: Decimal
    term_years: int
    insured_name: str
    insured_age: int
    status: str
    created_at: datetime
    paid_at: datetime | None = None

    @computed_field
    @property
    def status_text(self) -> str:
        return ORDER_STATUS_TEXT.get(self.status, self.status)


class OrderPaidOut(BaseModel):
    """支付成功后的返回：订单 + 同时生成的保单号。"""

    order: OrderOut
    policy_no: str
    policy_status: str
