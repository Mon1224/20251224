"""应用启动入口：注册路由与生命周期。

对应海报里的 app/main.py。
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.common.deps import AgentDB, BizDB
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging
from app.infrastructure.database import dispose_engines
from app.infrastructure.embedding import get_embedding_model
from app.infrastructure.milvus import close_milvus, get_milvus_client
from app.infrastructure.redis import close_redis, redis_client
from app.infrastructure.reranker import get_reranker
from app.modules.agent.graph import get_graph
from app.modules.agent.router import router as agent_router
from app.modules.claim.router import router as claim_router
from app.modules.conversation.router import router as conversation_router
from app.modules.order.router import router as order_router
from app.modules.policy.router import router as policy_router
from app.modules.product.router import router as product_router
from app.modules.user.router import router as user_router

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动与关闭时执行的收尾工作。

    yield 之前 = 启动时预热：建立 Milvus 连接、加载 embedding/重排模型、编译 Agent 图。
    这些资源默认都是惰性初始化，不预热的话第一次检索/对话要多等十几秒，
    而且依赖故障要拖到用户第一次提问才暴露。这里选择在启动阶段就做、失败即退出。
    yield 之后 = 关闭时执行，把 Milvus、Redis 连接和数据库连接池还回去。
    """
    logger.info("服务启动中：%s v%s", settings.app_name, settings.app_version)
    if settings.warmup_on_startup:
        # 模型加载是阻塞的 CPU/IO 操作，放到线程里跑，避免卡住事件循环。
        # 本地开发用 --reload 时可以把 WARMUP_ON_STARTUP 设为 false，改一次代码就不用重载模型。
        await asyncio.to_thread(_warm_up)
    else:
        logger.info("已关闭启动预热（WARMUP_ON_STARTUP=false），首次请求会承担模型加载耗时")
    yield
    logger.info("服务关闭中，正在释放连接...")
    await close_redis()
    close_milvus()
    await dispose_engines()
    logger.info("服务已安全关闭")


def _warm_up() -> None:
    """建立 Milvus 连接、加载向量/重排模型、编译 LangGraph 状态图。"""
    client = get_milvus_client()
    # MilvusClient 是惰性连接的，list_collections 会触发真正的 gRPC 连接，
    # 这样 Milvus 不可达时能在启动阶段就报错退出，而不是拖到第一次检索。
    client.list_collections()
    get_embedding_model()
    get_reranker()
    get_graph()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

# 允许前端（默认 Vite 跑在 5173 端口）跨域调用本服务
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 统一异常处理：业务异常 / 参数校验失败 / 未捕获异常
register_exception_handlers(app)

# 业务路由：按模块注册，新增模块时在这里加一行
for _router in (
    user_router,
    product_router,
    order_router,
    policy_router,
    claim_router,
    agent_router,
    conversation_router,
):
    app.include_router(_router)


@app.get("/health", tags=["系统"])
async def health() -> dict[str, str]:
    """最基础的健康检查：服务进程本身是否活着。"""
    return {"status": "ok"}


@app.get("/health/deps", tags=["系统"])
async def health_deps(biz_db: BizDB, agent_db: AgentDB) -> JSONResponse:
    """依赖健康检查：数据库和 Redis 到底通不通。

    全部正常返回 200；任一依赖不可用返回 503，
    这样编排系统（K8s / docker healthcheck / 负载均衡）能据此摘流量，
    而不是看到一个 200 就把请求继续打进来。
    """
    result: dict[str, dict[str, str]] = {}

    for name, session in (("insurance_biz", biz_db), ("insurance_agent", agent_db)):
        try:
            await session.execute(text("SELECT 1"))
            result[name] = {"status": "ok"}
        except Exception as exc:  # noqa: BLE001
            logger.error("数据库 %s 连接失败：%s", name, exc)
            result[name] = {"status": "error", "message": str(exc)}

    try:
        await redis_client.ping()
        result["redis"] = {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        logger.error("Redis 连接失败：%s", exc)
        result["redis"] = {"status": "error", "message": str(exc)}

    healthy = all(item["status"] == "ok" for item in result.values())
    return JSONResponse(status_code=200 if healthy else 503, content=result)
