"""全局配置：所有环境变量在这里集中读取，代码里不允许出现硬编码的地址和密钥。

对应海报里的 core/config.py。
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# app/core/config.py -> parents[2] 就是 agent-service 目录
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """从 .env 读取配置。字段名小写，环境变量名大写，两者自动对应。"""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env 里有多余的变量不报错
    )

    # ---------- 应用 ----------
    app_name: str = "安心保 · 保险 AI 智能顾问"
    app_version: str = "0.1.0"
    debug: bool = True
    # 启动时是否预热（连 Milvus、加载向量/重排模型、编译图）。
    # 生产保持 true：首个请求不用等模型加载，依赖故障也能启动即暴露。
    # 本地 --reload 开发时设为 false：改代码热重载不会每次重载模型。
    warmup_on_startup: bool = True

    # ---------- 数据库 ----------
    biz_db_url: str
    agent_db_url: str
    redis_url: str = "redis://localhost:6379/0"

    # 连接池参数：见第 4 步，这里是"借用/归还"策略的具体数值
    db_pool_size: int = 10          # 常驻连接数
    db_max_overflow: int = 20       # 高峰期允许临时多开的连接数
    db_pool_timeout: int = 30       # 借不到连接时的等待秒数
    db_pool_recycle: int = 1800     # 连接最长存活秒数，超过就回收重建
    db_echo: bool = False           # True 会打印所有 SQL，调试时再开

    # ---------- 向量库 ----------
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "insurance_clauses"

    # ---------- 向量化 ----------
    embedding_device: str = "cpu"   # 没有独立显卡就保持 cpu
    embedding_dim: int = 1024       # Qwen3-Embedding-0.6B 的输出维度
    embedding_batch_size: int = 16

    # ---------- 条款切分 ----------
    # 中文按"字数"近似控制切片长度：一个汉字大致等于一到两个 token，
    # 用字符数比引入完整分词器简单得多，对本项目足够。
    chunk_size: int = 500
    chunk_overlap: int = 80

    # ---------- 检索 ----------
    retrieval_top_k: int = 20        # 稠密/稀疏每路各召回多少候选
    retrieval_rrf_k: int = 60        # RRF 公式里的平滑常数
    retrieval_final_top_n: int = 5   # 融合重排后最终返回几条
    # 送进重排模型的候选数。这是整条链路上最慢的一环：
    # 纯 CPU 实测约 2 秒/条，6 条 ≈ 13 秒，10 条 ≈ 20 秒。质量上 6 条已经够用。
    retrieval_rerank_pool: int = 6
    enable_rerank: bool = True       # 关掉可降到约 4 秒，代价是 Top-1 命中率下降
    reranker_max_length: int = 512   # 我们的切片最长 178 字，512 足够

    # ---------- 大模型 ----------
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    llm_timeout: int = 60

    # ---------- 本地模型路径 ----------
    embedding_model_path: str = ""
    reranker_model_path: str = ""

    # ---------- 鉴权 ----------
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    # ---------- CORS ----------
    # 用逗号分隔的字符串，避免 pydantic 解析列表时踩坑
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """带缓存，整个进程只读一次 .env。"""
    return Settings()


settings = get_settings()
