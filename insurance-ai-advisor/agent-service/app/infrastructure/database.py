"""数据库接入与连接池。

对应海报里的 infrastructure/database.py，也是第 4 步的核心：
"通过连接池有效地管理数据库会话，避免重复连接/断开连接带来的额外开销"。

本项目有两个库，所以有两个 engine：
  - insurance_biz   : 传统保险业务数据
  - insurance_agent : 会话、消息、Agent 运行记录
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class BizBase(DeclarativeBase):
    """业务库（insurance_biz）所有 ORM 模型的基类。

    为什么分成两个 Base：因为业务库和智能库是两个独立数据库，
    Alembic 需要分别拿到各自的 metadata 才能生成正确的迁移脚本。
    """


class AgentBase(DeclarativeBase):
    """智能库（insurance_agent）所有 ORM 模型的基类。"""


def _build_engine(url: str) -> AsyncEngine:
    """按统一参数创建异步引擎，连接池策略集中在这里配置。"""
    return create_async_engine(
        url,
        echo=settings.db_echo,
        pool_size=settings.db_pool_size,        # 常驻连接数
        max_overflow=settings.db_max_overflow,  # 高峰期的临时连接上限
        pool_timeout=settings.db_pool_timeout,  # 借不到连接时等待多久后报错
        pool_recycle=settings.db_pool_recycle,  # 超时回收，防止被数据库端掐断
        pool_pre_ping=True,                     # 借用前探活，避免拿到失效连接
    )


# ---------- 业务库 ----------
biz_engine: AsyncEngine = _build_engine(settings.biz_db_url)
BizSessionLocal = async_sessionmaker(biz_engine, expire_on_commit=False, class_=AsyncSession)

# ---------- 智能库 ----------
agent_engine: AsyncEngine = _build_engine(settings.agent_db_url)
AgentSessionLocal = async_sessionmaker(agent_engine, expire_on_commit=False, class_=AsyncSession)


async def dispose_engines() -> None:
    """关闭时释放连接池，避免连接泄漏。由 lifespan 在 shutdown 阶段调用。"""
    await biz_engine.dispose()
    await agent_engine.dispose()


async def get_biz_session() -> AsyncGenerator[AsyncSession, None]:
    """业务库会话。请求结束自动归还连接到池子里。"""
    async with BizSessionLocal() as session:
        yield session


async def get_agent_session() -> AsyncGenerator[AsyncSession, None]:
    """智能库会话。"""
    async with AgentSessionLocal() as session:
        yield session
