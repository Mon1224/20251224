"""用户模块的请求与响应模型。

schemas 只负责"数据的形状和合法性"，不碰数据库、不写业务逻辑。
海报里写的"1. 前端请求模型 2. 前端响应模型"就是这两个方向。
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

# 中国大陆手机号：1 开头，第二位 3-9，共 11 位
PHONE_PATTERN = r"^1[3-9]\d{9}$"


class UserRegisterIn(BaseModel):
    """注册请求体。"""

    phone: str = Field(..., pattern=PHONE_PATTERN, description="手机号，也是登录账号")
    password: str = Field(..., min_length=8, max_length=64, description="密码，8-64 位")
    nickname: str | None = Field(None, max_length=50)
    real_name: str | None = Field(None, max_length=50)
    id_card_no: str | None = Field(None, max_length=32)


class UserLoginIn(BaseModel):
    """登录请求体。"""

    phone: str = Field(..., pattern=PHONE_PATTERN)
    password: str = Field(..., min_length=1, max_length=64)


class UserOut(BaseModel):
    """用户信息响应体。

    注意这里**没有** id_card_no：身份证号属于敏感信息，
    接口默认不返回，需要时单独提供脱敏后的接口。
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    phone: str
    nickname: str | None = None
    real_name: str | None = None
    is_active: bool
    created_at: datetime


class TokenOut(BaseModel):
    """登录/注册成功后的响应。"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="有效期（秒）")
    user: UserOut
