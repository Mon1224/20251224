"""大模型客户端：DeepSeek，走 OpenAI 兼容接口。

为什么用兼容接口而不是专用 SDK：
  以后换成 Qwen、Kimi、GPT 只需要改 .env 里的 base_url 和 model，
  业务代码一行都不用动。这是把"模型供应商"当成可替换部件的前提。
"""

import logging
from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def get_chat_model(temperature: float = 0.3, max_tokens: int = 1024) -> ChatOpenAI:
    """获取对话模型。

    temperature 的用法有讲究：
      0    —— 意图分类、参数抽取这类"要稳定"的任务
      0.3  —— 生成回答，保留一点表达灵活性但不会跑偏
    保险是强监管行业，不建议用 1.0 这种高随机性设置。
    """
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未配置，请在 .env 里填写")

    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=settings.llm_timeout,
        max_retries=2,
        # 流式模式下默认拿不到 token 用量，显式打开才会在最后一个 chunk 带上
        stream_usage=True,
    )
