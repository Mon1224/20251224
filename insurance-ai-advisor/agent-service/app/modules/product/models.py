"""保险产品、条款、条款切片三张表（insurance_biz）。"""

from decimal import Decimal

from sqlalchemy import BigInteger, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.mixins import TimestampMixin
from app.infrastructure.database import BizBase


class Product(TimestampMixin, BizBase):
    """保险产品。保费试算就是拿这张表里的规则算价。"""

    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(32), unique=True, index=True, comment="产品编码")
    name: Mapped[str] = mapped_column(String(100), comment="产品名称")
    company: Mapped[str] = mapped_column(String(100), comment="承保公司")
    category: Mapped[str] = mapped_column(
        String(32), index=True, comment="险种：重疾险/医疗险/意外险/寿险/年金险"
    )
    coverage_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), comment="参考保额（元）")
    premium_min: Mapped[Decimal] = mapped_column(Numeric(12, 2), comment="最低保费（元/年）")
    premium_max: Mapped[Decimal] = mapped_column(Numeric(12, 2), comment="最高保费（元/年）")
    base_rate: Mapped[Decimal] = mapped_column(
        Numeric(10, 6), comment="基础费率，保费试算用：保额 × 费率 × 年龄系数"
    )
    age_min: Mapped[int] = mapped_column(Integer, default=18, comment="最小投保年龄")
    age_max: Mapped[int] = mapped_column(Integer, default=60, comment="最大投保年龄")
    term_options: Mapped[str] = mapped_column(
        String(100), comment="可选保险期间，逗号分隔，如 10,20,30"
    )
    waiting_period_days: Mapped[int] = mapped_column(
        Integer, default=90, comment="等待期天数，条款问答的高频问题"
    )
    tags: Mapped[str | None] = mapped_column(String(200), comment="标签，逗号分隔，用于推荐")
    description: Mapped[str | None] = mapped_column(Text, comment="产品简介")
    status: Mapped[str] = mapped_column(
        String(20), default="on_sale", server_default="on_sale", comment="on_sale / off_shelf"
    )

    clauses: Mapped[list["Clause"]] = relationship(
        back_populates="product", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Product {self.code} {self.name}>"


class Clause(TimestampMixin, BizBase):
    """条款条目。一条 = 一个条款，例如「第 3 条 保险责任」。

    这是 RAG 的原始资料：ingest_clauses.py 会把每一条切分成若干 chunk 再向量化。
    """

    __tablename__ = "clauses"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE"), index=True
    )
    clause_no: Mapped[str] = mapped_column(String(32), comment="条款编号，如 3.2")
    title: Mapped[str] = mapped_column(String(200), comment="条款标题")
    content: Mapped[str] = mapped_column(Text, comment="条款正文")
    version: Mapped[str] = mapped_column(
        String(20), default="1.0", server_default="1.0", comment="条款版本"
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", comment="已切分并入库的片段数"
    )

    product: Mapped["Product"] = relationship(back_populates="clauses")
    chunks: Mapped[list["ClauseChunk"]] = relationship(
        back_populates="clause", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Clause {self.clause_no} {self.title}>"


class ClauseChunk(TimestampMixin, BizBase):
    """条款切片。每一行对应 Milvus 里的一个向量。

    为什么要单独一张表：切分后一个条款对应多个片段，
    检索命中后要能反查回"这句话出自哪个条款"，所以向量主键必须落库存档。
    """

    __tablename__ = "clause_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    clause_id: Mapped[int] = mapped_column(
        ForeignKey("clauses.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, comment="在本条款内的顺序，从 0 开始")
    content: Mapped[str] = mapped_column(Text, comment="切片正文，就是被向量化的原文")
    token_count: Mapped[int | None] = mapped_column(Integer, comment="切片长度（token 估算）")
    vector_id: Mapped[int | None] = mapped_column(
        BigInteger, index=True, comment="Milvus 中的向量主键，入库后回写"
    )

    clause: Mapped["Clause"] = relationship(back_populates="chunks")

    def __repr__(self) -> str:
        return f"<ClauseChunk clause={self.clause_id} #{self.chunk_index}>"
