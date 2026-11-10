"""产品模块的请求与响应模型。"""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class ClauseOut(BaseModel):
    """条款条目。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    clause_no: str
    title: str
    content: str


class ProductOut(BaseModel):
    """列表页用的产品摘要。字段刻意精简，避免列表接口返回几百 KB。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    code: str
    name: str
    company: str
    category: str
    coverage_amount: Decimal
    premium_min: Decimal
    premium_max: Decimal
    age_min: int
    age_max: int
    term_options: str
    waiting_period_days: int
    tags: str | None = None
    status: str


class ProductDetailOut(ProductOut):
    """详情页在产品摘要基础上多带简介和条款正文。"""

    description: str | None = None
    clauses: list[ClauseOut] = Field(default_factory=list)
    created_at: datetime


class PremiumCalcIn(BaseModel):
    """保费试算请求。"""

    product_id: int
    insured_age: int = Field(..., ge=0, le=100, description="被保人年龄")
    coverage_amount: Decimal = Field(..., gt=0, description="投保保额（元）")
    term_years: int = Field(..., gt=0, description="保险期间（年）")


class PremiumCalcOut(BaseModel):
    """保费试算结果。

    返回计算过程而不只是最终价格，因为第 10 步 AI 顾问要能向用户解释
    "为什么是这个价"，只给一个数字它没法回答。
    """

    product_id: int
    product_name: str
    insured_age: int
    coverage_amount: Decimal
    term_years: int

    base_premium: Decimal = Field(..., description="基础保费 = 保额 × 基础费率")
    age_factor: Decimal = Field(..., description="年龄系数")
    term_factor: Decimal = Field(..., description="期限系数")
    premium: Decimal = Field(..., description="最终保费（元/年）")
    breakdown: str = Field(..., description="人话版计算说明")
