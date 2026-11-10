"""ORM 公共字段混入，避免每张表都重复写 created_at / updated_at。"""

from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import Mapped, mapped_column


class TimestampMixin:
    """创建时间和更新时间。

    server_default=func.now() 交给数据库生成，即使绕过 ORM 直接写 SQL 也有值。
    onupdate=func.now() 让每次 UPDATE 自动刷新 updated_at。
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
