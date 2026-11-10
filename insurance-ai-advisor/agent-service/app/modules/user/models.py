"""用户表（insurance_biz.users）。"""

from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column

from app.common.mixins import TimestampMixin
from app.infrastructure.database import BizBase


class User(TimestampMixin, BizBase):
    """一个用户 = 一个手机号。

    实名信息（real_name / id_card_no）在投保时必填，注册时不强制。
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    phone: Mapped[str] = mapped_column(
        String(20), unique=True, index=True, comment="手机号，同时也是登录账号"
    )
    password_hash: Mapped[str] = mapped_column(
        String(255), comment="密码哈希值，绝不存明文密码"
    )
    nickname: Mapped[str | None] = mapped_column(String(50), comment="昵称")
    real_name: Mapped[str | None] = mapped_column(String(50), comment="真实姓名")
    id_card_no: Mapped[str | None] = mapped_column(String(32), comment="身份证号")
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default="true", comment="账号是否可用"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} phone={self.phone}>"
