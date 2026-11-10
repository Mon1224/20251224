"""Alembic 迁移环境。

本项目有两个数据库，用两个 ini 文件分别管理，共用这一个 env.py：
    alembic -c alembic.ini        ...  -> insurance_biz   业务库
    alembic -c alembic.agent.ini  ...  -> insurance_agent 智能库

靠 ini 里的 db_key 决定：连哪个库、比对哪份 metadata、版本记录写进哪张表。
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import settings
from app.registry import AGENT_METADATA, BIZ_METADATA

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

DATABASES = {
    "biz": {
        "url": settings.biz_db_url,
        "metadata": BIZ_METADATA,
        "version_table": "alembic_version_biz",
    },
    "agent": {
        "url": settings.agent_db_url,
        "metadata": AGENT_METADATA,
        "version_table": "alembic_version_agent",
    },
}

DB_KEY = config.get_main_option("db_key") or "biz"
if DB_KEY not in DATABASES:
    raise RuntimeError(f"未知的 db_key：{DB_KEY}，可选 {'/'.join(DATABASES)}")

db = DATABASES[DB_KEY]
target_metadata = db["metadata"]


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL 文件，不连数据库。alembic upgrade head --sql"""
    context.configure(
        url=db["url"],
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=db["version_table"],
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table=db["version_table"],
        compare_type=True,           # 字段类型变了也能检测到
        compare_server_default=True, # 默认值变了也能检测到
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = create_async_engine(db["url"], poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
