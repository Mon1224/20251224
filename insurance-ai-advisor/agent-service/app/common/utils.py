"""通用工具函数。"""

import secrets
from datetime import UTC, date, datetime


def _generate_no(prefix: str) -> str:
    """业务单号 = 前缀 + 时间戳 + 8 位随机码。

    为什么不用自增数字：单号会展示给用户，自增会让竞争对手推算你的业务量
    （比如"这家公司一天才出 30 单"）。加随机码同时也避免并发撞号。
    """
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    return f"{prefix}{stamp}{secrets.token_hex(4).upper()}"


def generate_order_no() -> str:
    return _generate_no("ORD")


def generate_policy_no() -> str:
    return _generate_no("POL")


def generate_claim_no() -> str:
    return _generate_no("CLM")


def add_years(start: date, years: int) -> date:
    """日期加年，处理 2 月 29 日这种闰年边界。"""
    try:
        return start.replace(year=start.year + years)
    except ValueError:
        return start.replace(year=start.year + years, day=28)


def parse_term_options(raw: str) -> list[int]:
    """把 "10,20,30" 这样的字符串转成 [10, 20, 30]。"""
    result: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            result.append(int(part))
    return result
