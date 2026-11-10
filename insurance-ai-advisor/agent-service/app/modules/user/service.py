"""用户模块的业务逻辑层。

这一层负责"编排"：判断手机号是否重复、校验密码、签发令牌。
它不直接写 SQL（交给 repository），也不处理 HTTP（交给 router）。
"""

import logging

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AuthError, BusinessError
from app.core.security import create_access_token, hash_password, verify_password
from app.modules.user.models import User
from app.modules.user.repository import UserRepository
from app.modules.user.schemas import UserLoginIn, UserRegisterIn

logger = logging.getLogger(__name__)


class UserService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repo = UserRepository(session)

    async def register(self, payload: UserRegisterIn) -> tuple[User, str, int]:
        """注册并直接返回登录令牌，省掉用户再登录一次。"""
        if await self.repo.get_by_phone(payload.phone):
            raise BusinessError("该手机号已注册，请直接登录", code="PHONE_ALREADY_EXISTS")

        try:
            user = await self.repo.create(
                phone=payload.phone,
                password_hash=hash_password(payload.password),
                nickname=payload.nickname,
                real_name=payload.real_name,
                id_card_no=payload.id_card_no,
            )
        except IntegrityError:
            # 两个请求同时注册同一个手机号时，先到的那个能成功，
            # 后到的会撞上数据库的唯一索引。这里把它翻译成人话。
            raise BusinessError("该手机号已注册，请直接登录", code="PHONE_ALREADY_EXISTS") from None

        await self.session.commit()
        await self.session.refresh(user)

        token, expires_in = create_access_token(user.id)
        logger.info("新用户注册成功 user_id=%s", user.id)
        return user, token, expires_in

    async def login(self, payload: UserLoginIn) -> tuple[User, str, int]:
        user = await self.repo.get_by_phone(payload.phone)

        # 注意：手机号不存在和密码错误返回同一句提示，
        # 避免攻击者通过错误信息判断某个手机号是否注册过。
        if user is None or not verify_password(payload.password, user.password_hash):
            raise AuthError("手机号或密码错误")

        if not user.is_active:
            raise BusinessError("账号已被停用，请联系客服", code="USER_DISABLED")

        token, expires_in = create_access_token(user.id)
        logger.info("用户登录成功 user_id=%s", user.id)
        return user, token, expires_in
