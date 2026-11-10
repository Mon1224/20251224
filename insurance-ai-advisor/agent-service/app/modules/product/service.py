"""产品模块的业务逻辑层。

这里的核心是「保费试算」：纯规则计算，不是 AI。
它同时被接口和 Agent 工具（calculate_premium）复用，AI 能"帮用户算保费"。
"""

from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.common.utils import parse_term_options
from app.core.exceptions import BusinessError, NotFoundError
from app.modules.product.models import Product
from app.modules.product.repository import ProductRepository
from app.modules.product.schemas import PremiumCalcIn, PremiumCalcOut

# 年龄系数：年龄越大风险越高，保费越贵
AGE_FACTORS: list[tuple[int, Decimal]] = [
    (25, Decimal("0.80")),
    (35, Decimal("1.00")),
    (45, Decimal("1.30")),
    (55, Decimal("1.70")),
    (200, Decimal("2.20")),
]

# 期限系数：保障期越长，保险公司承担风险的时间越久
TERM_FACTORS: dict[int, Decimal] = {
    1: Decimal("1.00"),
    10: Decimal("0.90"),
    15: Decimal("0.95"),
    20: Decimal("1.00"),
    30: Decimal("1.15"),
}
DEFAULT_TERM_FACTOR = Decimal("1.00")

MONEY = Decimal("0.01")


def _age_factor(age: int) -> Decimal:
    for limit, factor in AGE_FACTORS:
        if age <= limit:
            return factor
    return AGE_FACTORS[-1][1]


def _term_factor(years: int) -> Decimal:
    return TERM_FACTORS.get(years, DEFAULT_TERM_FACTOR)


def calc_premium(
    product: Product, insured_age: int, coverage_amount: Decimal, term_years: int
) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    """算保费，返回 (最终保费, 基础保费, 年龄系数, 期限系数)。

    公式：最终保费 = 保额 × 基础费率 × 年龄系数 × 期限系数
    注意全程用 Decimal 而不是 float——金额计算用浮点数会丢精度，
    0.1 + 0.2 != 0.3 这个问题在保费上就是真金白银的差错。
    """
    base = coverage_amount * product.base_rate
    age_f = _age_factor(insured_age)
    term_f = _term_factor(term_years)
    premium = (base * age_f * term_f).quantize(MONEY, rounding=ROUND_HALF_UP)
    return premium, base.quantize(MONEY, rounding=ROUND_HALF_UP), age_f, term_f


class ProductService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repo = ProductRepository(session)

    async def list_products(
        self, *, category: str | None, keyword: str | None, page: int, page_size: int
    ) -> tuple[list[Product], int]:
        return await self.repo.list_products(
            category=category, keyword=keyword, page=page, page_size=page_size
        )

    async def get_detail(self, product_id: int) -> Product:
        product = await self.repo.get_by_id(product_id, with_clauses=True)
        if product is None:
            raise NotFoundError("产品不存在或已下架")
        return product

    async def calculate(self, payload: PremiumCalcIn) -> PremiumCalcOut:
        product = await self.repo.get_by_id(payload.product_id)
        if product is None:
            raise NotFoundError("产品不存在或已下架")

        if not (product.age_min <= payload.insured_age <= product.age_max):
            raise BusinessError(
                f"该产品投保年龄为 {product.age_min}-{product.age_max} 岁，"
                f"被保人 {payload.insured_age} 岁不符合条件",
                code="AGE_NOT_ALLOWED",
            )

        allowed_terms = parse_term_options(product.term_options)
        if allowed_terms and payload.term_years not in allowed_terms:
            raise BusinessError(
                f"该产品可选保险期间为 {product.term_options} 年，"
                f"不支持 {payload.term_years} 年",
                code="TERM_NOT_ALLOWED",
            )

        premium, base, age_f, term_f = calc_premium(
            product, payload.insured_age, payload.coverage_amount, payload.term_years
        )

        breakdown = (
            f"基础保费 = 保额 {payload.coverage_amount} × 费率 {product.base_rate} = {base} 元；"
            f"年龄系数 {age_f}（{payload.insured_age} 岁）；"
            f"期限系数 {term_f}（{payload.term_years} 年）；"
            f"最终保费 = {base} × {age_f} × {term_f} = {premium} 元/年"
        )

        return PremiumCalcOut(
            product_id=product.id,
            product_name=product.name,
            insured_age=payload.insured_age,
            coverage_amount=payload.coverage_amount,
            term_years=payload.term_years,
            base_premium=base,
            age_factor=age_f,
            term_factor=term_f,
            premium=premium,
            breakdown=breakdown,
        )
