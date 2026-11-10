"""公共依赖：接口需要什么，从这里注入，不用自己 new。

对应海报里的 common/deps.py。
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AuthError
from app.core.security import decode_access_token, is_token_revoked
from app.infrastructure.database import get_agent_session, get_biz_session
from app.modules.user.models import User
from app.modules.user.repository import UserRepository

# 用法：async def list_products(db: BizDB): ...
BizDB = Annotated[AsyncSession, Depends(get_biz_session)]
AgentDB = Annotated[AsyncSession, Depends(get_agent_session)]

# auto_error=False：没带 Token 时不让 FastAPI 直接抛 403，
# 而是交给下面的函数抛我们自己的 AuthError，返回统一的错误结构。
bearer_scheme = HTTPBearer(auto_error=False)


@dataclass
class AuthContext:
    """当前登录用户的上下文，顺带带上令牌自身的信息。"""

    user: User
    jti: str | None
    expires_in: int  # 距离过期还有多少秒，退出登录时要用


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: BizDB,
) -> AuthContext:
    """从 Authorization: Bearer <token> 中解析出当前用户。

    任何一步不对都抛 AuthError（401），由全局异常处理器统一包装。
    """
    if credentials is None or not credentials.credentials:
        raise AuthError("请先登录")

    try:
        payload = decode_access_token(credentials.credentials)
    except JWTError:
        raise AuthError("登录已过期，请重新登录") from None

    user_id = payload.get("sub")
    jti = payload.get("jti")
    if not user_id:
        raise AuthError("令牌内容不合法")

    if jti and await is_token_revoked(jti):
        raise AuthError("登录已失效，请重新登录")

    user = await UserRepository(db).get_by_id(int(user_id))
    if user is None:
        raise AuthError("用户不存在")
    if not user.is_active:
        raise AuthError("账号已被停用，请联系客服")

    remaining = int(payload.get("exp", 0)) - int(datetime.now(UTC).timestamp())
    return AuthContext(user=user, jti=jti, expires_in=max(remaining, 1))


CurrentAuth = Annotated[AuthContext, Depends(get_current_user)]
CurrentUser = Annotated[User, Depends(get_current_user)]
