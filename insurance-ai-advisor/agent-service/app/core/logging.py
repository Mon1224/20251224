"""日志配置：系统的行车记录仪，出问题靠它复盘的。

对应海报里的 core/logging.py。
"""

import logging
import sys

from app.core.config import settings


def setup_logging() -> None:
    """初始化根日志器，应用启动时调用一次即可。"""
    level = logging.DEBUG if settings.debug else logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.handlers.clear()  # 避免 uvicorn --reload 重复挂载导致日志打印两遍
    root.addHandler(handler)
    root.setLevel(level)

    # 这些库在 DEBUG 级别下会把每条 SQL、每个 HTTP 请求、每个 SSE 分片都打出来，
    # 日志会被淹没到没法看。统一压到 WARNING。
    for noisy in (
        "sqlalchemy.engine",
        "httpx",
        "httpx2",
        "httpcore",
        "httpcore2",
        "openai",
        "sse_starlette",
        "sentence_transformers",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
