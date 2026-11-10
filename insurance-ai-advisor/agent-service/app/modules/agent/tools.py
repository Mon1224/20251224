"""Agent 能调用的业务工具。

这一层是"AI 顾问"和"聊天玩具"的分界线：
没有工具，模型只能凭训练数据空谈；有了工具，它能真的去查产品、算保费、读保单。

每个工具都做了两件事：
  1. 自己开数据库会话（工具是独立执行的，不共享接口的会话）
  2. 捕获异常并返回友好提示——工具报错不该让整轮对话崩掉。
     把错误信息交给模型，它反而能告诉用户"查询失败了，请稍后再试"。
"""

import json
import logging
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.core.exceptions import AppException
from app.infrastructure.database import BizSessionLocal
from app.modules.agent.retriever import search_clauses_async
from app.modules.claim.repository import ClaimRepository
from app.modules.order.repository import OrderRepository
from app.modules.policy.repository import PolicyRepository
from app.modules.product.repository import ProductRepository
from app.modules.product.schemas import PremiumCalcIn
from app.modules.product.service import ProductService

logger = logging.getLogger(__name__)

# 工具结果会全部塞进模型上下文，所以要控制体量，不能把整张表倒进去
MAX_ITEMS = 8
MAX_TEXT = 120


def _dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _brief(text: str | None, limit: int = MAX_TEXT) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + "…"


@tool
async def list_insurance_products(category: str = "", keyword: str = "") -> str:
    """查询在售的保险产品列表。用户想了解有哪些产品、或需要产品推荐时使用。

    Args:
        category: 险种筛选，只能是这几个值之一：重疾险、医疗险、意外险、寿险、年金险。
            不筛选就传空字符串。
        keyword: 关键词，会匹配产品名、承保公司和标签，例如"少儿""百万""养老"。不搜索就传空字符串。
    """
    try:
        async with BizSessionLocal() as session:
            items, total = await ProductRepository(session).list_products(
                category=category or None, keyword=keyword or None, page=1, page_size=MAX_ITEMS
            )
            return _dump(
                {
                    "total": total,
                    "products": [
                        {
                            "id": p.id,
                            "code": p.code,
                            "name": p.name,
                            "company": p.company,
                            "category": p.category,
                            "coverage_amount": str(p.coverage_amount),
                            "premium_range": f"{p.premium_min}~{p.premium_max} 元/年",
                            "age_range": f"{p.age_min}-{p.age_max} 岁",
                            "term_options": p.term_options,
                            "waiting_period_days": p.waiting_period_days,
                            "tags": p.tags,
                        }
                        for p in items
                    ],
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("查询产品失败")
        return _dump({"error": f"查询产品失败：{exc}"})


@tool
async def calculate_premium(
    product: str, insured_age: int, coverage_amount: float, term_years: int
) -> str:
    """试算保费。用户问"多少钱""贵不贵""帮我算一下"时使用。

    Args:
        product: 产品编码或产品名的一部分，例如 "XB-CI-001" 或 "康健终身重疾险"。
        insured_age: 被保人年龄，整数。
        coverage_amount: 投保保额，单位元，例如 500000 表示 50 万。
        term_years: 保险期间，单位年。不确定时先查产品看 term_options 再算。
    """
    try:
        async with BizSessionLocal() as session:
            repo = ProductRepository(session)
            target = await repo.get_by_code(product)
            if target is None:
                candidates, _ = await repo.list_products(keyword=product, page=1, page_size=1)
                target = candidates[0] if candidates else None
            if target is None:
                return _dump(
                    {"error": f"没有找到匹配「{product}」的产品，请先查询产品列表确认名称"}
                )

            result = await ProductService(session).calculate(
                PremiumCalcIn(
                    product_id=target.id,
                    insured_age=insured_age,
                    coverage_amount=coverage_amount,
                    term_years=term_years,
                )
            )
            return _dump(result.model_dump(mode="json"))
    except AppException as exc:
        # 超龄、期限不支持等是可预期的业务异常，直接把原因告诉模型
        return _dump({"error": exc.message, "code": exc.code})
    except Exception as exc:  # noqa: BLE001
        logger.exception("保费试算失败")
        return _dump({"error": f"保费试算失败：{exc}"})


@tool
async def get_my_policies(state: Annotated[dict, InjectedState]) -> str:
    """查询当前登录用户自己的保单。用户问"我的保单""什么时候到期""保额多少"时使用。

    不需要传用户参数，系统会自动注入当前登录用户。
    """
    try:
        async with BizSessionLocal() as session:
            items, total = await PolicyRepository(session).list_by_user(
                int(state["user_id"]), page=1, page_size=MAX_ITEMS
            )
            return _dump(
                {
                    "total": total,
                    "policies": [
                        {
                            "policy_no": p.policy_no,
                            "product_name": p.product_name,
                            "holder_name": p.holder_name,
                            "insured_name": p.insured_name,
                            "coverage_amount": str(p.coverage_amount),
                            "premium": str(p.premium),
                            "start_date": str(p.start_date),
                            "end_date": str(p.end_date),
                            "status": p.status,
                        }
                        for p in items
                    ],
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("查询保单失败")
        return _dump({"error": f"查询保单失败：{exc}"})


@tool
async def get_my_claims(state: Annotated[dict, InjectedState]) -> str:
    """查询当前登录用户的理赔记录与进度。用户问"我的理赔到哪一步了""赔了多少"时使用。

    不需要传用户参数，系统会自动注入当前登录用户。
    """
    try:
        async with BizSessionLocal() as session:
            items, total = await ClaimRepository(session).list_by_user(
                int(state["user_id"]), page=1, page_size=MAX_ITEMS
            )
            return _dump(
                {
                    "total": total,
                    "claims": [
                        {
                            "claim_no": c.claim_no,
                            "policy_no": c.policy_no,
                            "claim_type": c.claim_type,
                            "amount": str(c.amount),
                            "status": c.status,
                            "submitted_at": str(c.submitted_at) if c.submitted_at else None,
                            "description": _brief(c.description, 60),
                        }
                        for c in items
                    ],
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("查询理赔失败")
        return _dump({"error": f"查询理赔失败：{exc}"})


@tool
async def get_my_orders(state: Annotated[dict, InjectedState]) -> str:
    """查询当前登录用户的投保订单。用户问"我买过什么""订单支付了吗"时使用。"""
    try:
        async with BizSessionLocal() as session:
            items, total = await OrderRepository(session).list_by_user(
                int(state["user_id"]), page=1, page_size=MAX_ITEMS
            )
            return _dump(
                {
                    "total": total,
                    "orders": [
                        {
                            "order_no": o.order_no,
                            "product_name": o.product_name,
                            "premium": str(o.premium),
                            "insured_name": o.insured_name,
                            "status": o.status,
                            "created_at": str(o.created_at),
                        }
                        for o in items
                    ],
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("查询订单失败")
        return _dump({"error": f"查询订单失败：{exc}"})


@tool
async def search_insurance_clauses(query: str, product_id: int = 0) -> str:
    """检索保险条款原文。需要回答条款细节（等待期、免赔额、保障范围、责任免除等）时使用。

    Args:
        query: 要检索的问题或关键词，用自然语言描述即可。
        product_id: 可选，限定在某个产品内检索。不限定就传 0。
    """
    try:
        result = await search_clauses_async(
            query,
            mode="hybrid",
            final_top_n=3,
            product_id=product_id if product_id else None,
        )
        return _dump(
            {
                "hits": [
                    {
                        "clause_id": h.clause_id,
                        "product_name": h.product_name,
                        "clause_no": h.clause_no,
                        "content": _brief(h.content, 300),
                    }
                    for h in result.hits
                ]
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("条款检索失败")
        return _dump({"error": f"条款检索失败：{exc}"})


BUSINESS_TOOLS = [
    list_insurance_products,
    calculate_premium,
    get_my_policies,
    get_my_claims,
    get_my_orders,
    search_insurance_clauses,
]
