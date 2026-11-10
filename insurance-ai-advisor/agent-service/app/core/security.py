"""安全工具：密码哈希、JWT 签发校验、Token 吊销名单。

三条不能破的规矩：
  1. 数据库里永远不存明文密码，只存哈希
  2. JWT 的签名密钥从 .env 读，不进代码库
  3. 退出登录要把 Token 拉黑，否则它在过期前一直有效
"""

import logging
import uuid
from datetime import UTC, datetime, timedelta

import bcrypt
from jose import jwt

from app.core.config import settings

logger = logging.getLogger(__name__)

# bcrypt 只处理密码的前 72 字节，超长部分会被忽略。
# 我们在 schemas 里把密码上限压到 64 个字符，从源头避开这个坑。
BCRYPT_MAX_BYTES = 72


# ---------------------------------------------------------------- 密码
def hash_password(raw_password: str) -> str:
    """把明文密码变成哈希。盐值由 bcrypt 自动生成并写进结果字符串。"""
    return bcrypt.hashpw(raw_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(raw_password: str, password_hash: str) -> bool:
    """校验密码。哈希损坏或无效应返回 False，而不是抛异常。"""
    try:
        return bcrypt.checkpw(raw_password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------- JWT
def create_access_token(user_id: int, expires_minutes: int | None = None) -> tuple[str, int]:
    """签发访问令牌，返回 (token, 有效秒数)。

    sub  : 令牌主体，这里放用户 id
    jti  : 令牌唯一编号，退出登录时用它把令牌加入黑名单
    exp  : 过期时间，JWT 库会自己校验
    """
    minutes = expires_minutes or settings.jwt_expire_minutes
    now = datetime.now(UTC)
    expire = now + timedelta(minutes=minutes)

    payload = {
        "sub": str(user_id),
        "jti": uuid.uuid4().hex,
        "iat": now,
        "exp": expire,
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, minutes * 60


def decode_access_token(token: str) -> dict:
    """解析并校验令牌。签名不对或已过期都会抛 JWTError，由调用方处理。"""
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])


# ---------------------------------------------------------------- 黑名单
def _blacklist_key(jti: str) -> str:
    return f"jwt:blacklist:{jti}"


async def revoke_token(jti: str, expires_in: int) -> None:
    """把令牌加入黑名单，过期时间一到 Redis 自动删除，不会无限增长。

    与 is_token_revoked 保持同一套故障策略：Redis 不可用时记警告而不是抛异常。
    否则 Redis 一抖动，用户点"退出登录"就会收到 500。代价是短期内该令牌
    仍有效——与"读取侧放行"同源，统一按"黑名单是安全增强、非核心链路"处理。
    """
    from app.infrastructure.redis import redis_client

    try:
        await redis_client.setex(_blacklist_key(jti), max(expires_in, 1), "1")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis 不可用，Token 加入黑名单失败：%s", exc)


async def is_token_revoked(jti: str) -> bool:
    """检查令牌是否已被吊销。

    Redis 挂了怎么办？这里选择"放行 + 记警告"。
    因为吊销名单是安全增强而非核心链路，Redis 故障时让所有用户都登不上，
    代价比短期容忍已注销的 Token 更大。生产环境要按自己的风险偏好决定。
    """
    from app.infrastructure.redis import redis_client

    try:
        return await redis_client.exists(_blacklist_key(jti)) == 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis 不可用，跳过 Token 黑名单检查：%s", exc)
        return False
