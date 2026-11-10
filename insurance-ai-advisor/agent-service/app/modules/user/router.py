"""用户模块的接口层。

router 只做三件事：定义地址、接收参数、返回响应。
业务判断一律交给 service，所以这里的函数都很短。
"""

from fastapi import APIRouter, status

from app.common.deps import BizDB, CurrentAuth
from app.core.exceptions import AuthError
from app.core.security import revoke_token
from app.modules.user.schemas import TokenOut, UserLoginIn, UserOut, UserRegisterIn
from app.modules.user.service import UserService

router = APIRouter(prefix="/api/v1/auth", tags=["用户与鉴权"])


@router.post(
    "/register",
    response_model=TokenOut,
    status_code=status.HTTP_201_CREATED,
    summary="注册",
)
async def register(payload: UserRegisterIn, db: BizDB) -> TokenOut:
    user, token, expires_in = await UserService(db).register(payload)
    return TokenOut(access_token=token, expires_in=expires_in, user=UserOut.model_validate(user))


@router.post("/login", response_model=TokenOut, summary="登录")
async def login(payload: UserLoginIn, db: BizDB) -> TokenOut:
    user, token, expires_in = await UserService(db).login(payload)
    return TokenOut(access_token=token, expires_in=expires_in, user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut, summary="获取当前登录用户")
async def me(auth: CurrentAuth) -> UserOut:
    return UserOut.model_validate(auth.user)


@router.post("/logout", summary="退出登录")
async def logout(auth: CurrentAuth) -> dict[str, str]:
    """把当前 Token 加入黑名单。

    JWT 是无状态的，签发之后服务端本来管不了它。
    所以退出登录时把它的唯一编号写进 Redis，后续请求就会被拒。
    """
    if auth.jti is None:
        raise AuthError("当前令牌缺少编号，无法注销")
    await revoke_token(auth.jti, auth.expires_in)
    return {"message": "已退出登录"}
