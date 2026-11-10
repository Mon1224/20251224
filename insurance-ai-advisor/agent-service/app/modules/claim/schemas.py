"""理赔模块的请求与响应模型。"""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, computed_field

CLAIM_STATUS_TEXT = {
    "submitted": "已提交",
    "reviewing": "审核中",
    "approved": "审核通过",
    "rejected": "已拒赔",
    "paid": "已赔付",
}


class ClaimCreateIn(BaseModel):
    policy_no: str = Field(..., description="理赔依据：保单号")
    claim_type: str = Field(..., max_length=32, description="出险类型：门诊/住院/重疾/意外")
    amount: Decimal = Field(..., gt=0, description="申请理赔金额（元）")
    description: str | None = Field(None, max_length=500, description="出险描述")


class ClaimOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    claim_no: str
    policy_no: str | None = None
    claim_type: str
    amount: Decimal
    approved_amount: Decimal | None = None
    description: str | None = None
    status: str
    submitted_at: datetime | None = None

    @computed_field
    @property
    def status_text(self) -> str:
        return CLAIM_STATUS_TEXT.get(self.status, self.status)
