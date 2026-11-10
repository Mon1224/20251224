"""统一的分页响应结构。

所有列表接口都返回同样的形状，前端只写一次分页组件就够了。
"""

import math
from typing import Generic, TypeVar

from pydantic import BaseModel, computed_field

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int

    @computed_field
    @property
    def pages(self) -> int:
        """总页数。total=0 时返回 1，前端不用处理边界。"""
        if self.page_size <= 0:
            return 1
        return max(1, math.ceil(self.total / self.page_size))
