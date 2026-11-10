"""保单模块的请求与响应模型。"""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, computed_field

POLICY_STATUS_TEXT = {
    "pending": "待生效",
    "active": "生效中",
    "expired": "已失效",
    "claimed": "已理赔",
    "surrendered": "已退保",
}


class PolicyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    policy_no: str
    product_id: int
    product_name: str | None = None
    holder_name: str
    insured_name: str
    beneficiary: str
    coverage_amount: Decimal
    premium: Decimal
    start_date: date
    end_date: date
    status: str

    @computed_field
    @property
    def status_text(self) -> str:
        return POLICY_STATUS_TEXT.get(self.status, self.status)
