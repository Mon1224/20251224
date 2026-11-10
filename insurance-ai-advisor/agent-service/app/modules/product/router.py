"""产品模块的接口层。"""

from fastapi import APIRouter, Query

from app.common.deps import BizDB
from app.common.pagination import Page
from app.modules.product.schemas import (
    PremiumCalcIn,
    PremiumCalcOut,
    ProductDetailOut,
    ProductOut,
)
from app.modules.product.service import ProductService

router = APIRouter(prefix="/api/v1/products", tags=["保险产品"])


@router.get("", response_model=Page[ProductOut], summary="产品列表（支持筛选与分页）")
async def list_products(
    db: BizDB,
    category: str | None = Query(None, description="险种：重疾险/医疗险/意外险/寿险/年金险"),
    keyword: str | None = Query(None, description="按名称、公司或标签模糊搜索"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> Page[ProductOut]:
    items, total = await ProductService(db).list_products(
        category=category, keyword=keyword, page=page, page_size=page_size
    )
    return Page[ProductOut](
        items=[ProductOut.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/premium", response_model=PremiumCalcOut, summary="保费试算")
async def calc_premium(payload: PremiumCalcIn, db: BizDB) -> PremiumCalcOut:
    """具体路径要写在参数路径前面，否则 /premium 会被当成 product_id 匹配掉。"""
    return await ProductService(db).calculate(payload)


@router.get("/{product_id}", response_model=ProductDetailOut, summary="产品详情（含条款）")
async def get_product(product_id: int, db: BizDB) -> ProductDetailOut:
    product = await ProductService(db).get_detail(product_id)
    return ProductDetailOut.model_validate(product)
