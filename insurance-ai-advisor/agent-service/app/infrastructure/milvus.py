"""Milvus 向量库客户端与集合管理。

集合结构（collection 相当于关系库里的"表"）：
    id            主键，Milvus 自动生成
    clause_id     回指业务库 clauses.id
    product_id    回指业务库 products.id
    chunk_index   这条切片在条款内的顺序
    clause_no     条款编号，检索结果直接展示，省一次回表
    product_name  产品名称，同上
    content       切片正文，也是 BM25 的输入字段
    sparse        稀疏向量 —— 由 Milvus 的 BM25 函数自动生成，插入时不用给
    dense         稠密向量 —— 由外部 embedding 模型生成，1024 维
"""

import logging

from pymilvus import DataType, Function, FunctionType, MilvusClient

from app.core.config import settings

logger = logging.getLogger(__name__)


# 进程内共享一个客户端：MilvusClient 底层是长连接（gRPC channel），
# 每次检索都 new 一个会造成连接反复建立/销毁。生命周期由 main.lifespan 统一管理。
_client: MilvusClient | None = None


def get_milvus_client() -> MilvusClient:
    """返回进程内共享的 Milvus 客户端，首次调用时惰性创建。"""
    global _client
    if _client is None:
        _client = MilvusClient(uri=settings.milvus_uri)
    return _client


def close_milvus() -> None:
    """关闭共享的 Milvus 客户端，释放底层连接。由 lifespan 在 shutdown 阶段调用。"""
    global _client
    if _client is not None:
        _client.close()
        _client = None


def build_schema():
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("clause_id", DataType.INT64)
    schema.add_field("product_id", DataType.INT64)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("clause_no", DataType.VARCHAR, max_length=64)
    schema.add_field("title", DataType.VARCHAR, max_length=128)
    schema.add_field("product_name", DataType.VARCHAR, max_length=512)
    # content 只存条款正文，用于展示/引用，不再参与检索。
    schema.add_field("content", DataType.VARCHAR, max_length=4096)
    # search_text = 产品名 + 条款标题 + 正文，dense 和 BM25 都基于它。
    # 只把正文做向量会丢两个强信号：用户问题里常带产品名和标题词，
    # 而正文里往往不出现这些词（多个产品的"责任免除"正文高度相似，难以区分）。
    #
    # enable_analyzer=True 让 Milvus 对这个字段做分词，BM25 才能算出稀疏向量。
    # analyzer_params 必须显式指定 jieba，否则 Milvus 默认分词器会把一整串连续中文
    # 当成【一个】token，导致查"等待期"命中 0 条，中文 BM25 等于废掉。
    schema.add_field(
        "search_text",
        DataType.VARCHAR,
        max_length=2048,
        enable_analyzer=True,
        analyzer_params={"tokenizer": "jieba", "filter": ["lowercase"]},
    )
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=settings.embedding_dim)

    # BM25 函数：插入数据时 Milvus 自动用 search_text 生成 sparse 向量，
    # 我们不需要额外跑 BM25 模型，也不用自己算稀疏向量。
    schema.add_function(
        Function(
            name="search_text_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["search_text"],
            output_field_names=["sparse"],
        )
    )
    return schema


def build_index_params():
    index_params = MilvusClient.prepare_index_params()
    # 稠密向量用 HNSW：召回率高、延迟低，适合这种中小规模数据
    index_params.add_index(
        field_name="dense",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    # 稀疏向量用倒排索引，度量就是 BM25 本身
    index_params.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )
    return index_params


def create_collection(client: MilvusClient, *, rebuild: bool = False) -> None:
    name = settings.milvus_collection
    if client.has_collection(name):
        if not rebuild:
            logger.info("集合 %s 已存在，跳过创建", name)
            return
        logger.warning("删除并重建集合 %s", name)
        client.drop_collection(name)

    client.create_collection(
        collection_name=name,
        schema=build_schema(),
        index_params=build_index_params(),
    )
    client.load_collection(name)
    logger.info("集合 %s 创建完成并已加载到内存", name)


def drop_collection(client: MilvusClient) -> None:
    name = settings.milvus_collection
    if client.has_collection(name):
        client.drop_collection(name)
        logger.info("已删除集合 %s", name)


def insert_chunks(client: MilvusClient, rows: list[dict]) -> list[int]:
    """插入切片，返回 Milvus 自动生成的主键列表（顺序与入参一致）。

    注意 rows 里不要带 sparse 字段——它由 BM25 函数在服务端自动生成。
    """
    result = client.insert(collection_name=settings.milvus_collection, data=rows)
    ids = list(result.get("ids", []))
    if len(ids) != len(rows):
        raise RuntimeError(f"Milvus 返回的主键数量不匹配：{len(ids)} != {len(rows)}")
    return ids


def collection_stats(client: MilvusClient) -> dict:
    name = settings.milvus_collection
    if not client.has_collection(name):
        return {"exists": False, "row_count": 0}
    stats = client.get_collection_stats(name)
    return {"exists": True, "row_count": int(stats.get("row_count", 0))}
