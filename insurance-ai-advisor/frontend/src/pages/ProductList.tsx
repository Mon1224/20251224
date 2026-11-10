import { Card, Col, Empty, Input, Pagination, Row, Select, Space, Spin, Tag } from "antd";
import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { productApi } from "@/api/endpoints";
import type { Product } from "@/api/types";

const CATEGORIES = ["重疾险", "医疗险", "意外险", "寿险", "年金险"];
const PAGE_SIZE = 6;

const money = (value: string) => Number(value).toLocaleString("zh-CN");

export default function ProductList() {
  const navigate = useNavigate();
  const [items, setItems] = useState<Product[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [category, setCategory] = useState<string>();
  const [keyword, setKeyword] = useState("");
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const result = await productApi.list({
        category,
        keyword: keyword || undefined,
        page,
        page_size: PAGE_SIZE,
      });
      setItems(result.items);
      setTotal(result.total);
    } finally {
      setLoading(false);
    }
  }, [category, keyword, page]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <>
      <h1 className="page-title">保险商城</h1>
      <div className="page-subtitle">
        挑一份合适的保障，右下角的 AI 顾问可以随时帮你解读条款
      </div>

      <Space style={{ marginBottom: 20 }} wrap>
        <Select
          allowClear
          placeholder="全部险种"
          style={{ width: 160 }}
          value={category}
          onChange={(value) => {
            setCategory(value);
            setPage(1);
          }}
          options={CATEGORIES.map((item) => ({ value: item, label: item }))}
        />
        <Input.Search
          placeholder="搜索产品名、公司或标签"
          style={{ width: 300 }}
          allowClear
          onSearch={(value) => {
            setKeyword(value.trim());
            setPage(1);
          }}
        />
      </Space>

      <Spin spinning={loading}>
        {items.length === 0 && !loading ? (
          <Empty description="没有找到匹配的产品" style={{ padding: "60px 0" }} />
        ) : (
          <Row gutter={[16, 16]}>
            {items.map((product) => (
              <Col key={product.id} xs={24} sm={12} lg={8}>
                <Card
                  hoverable
                  onClick={() => navigate(`/products/${product.id}`)}
                  title={product.name}
                  extra={<Tag color="cyan">{product.category}</Tag>}
                >
                  <div style={{ color: "#6b7280", fontSize: 13, marginBottom: 10 }}>
                    {product.company}
                  </div>
                  <div style={{ fontSize: 22, color: "#0f766e", fontWeight: 600 }}>
                    ¥{money(product.premium_min)}
                    <span style={{ fontSize: 13, color: "#6b7280", fontWeight: 400 }}>
                      {" "}
                      起 / 年
                    </span>
                  </div>
                  <div style={{ fontSize: 12.5, color: "#6b7280", marginTop: 10, lineHeight: 1.9 }}>
                    <div>保额 {money(product.coverage_amount)} 元</div>
                    <div>
                      投保年龄 {product.age_min}-{product.age_max} 岁
                    </div>
                    <div>
                      等待期{" "}
                      {product.waiting_period_days === 0
                        ? "无"
                        : `${product.waiting_period_days} 天`}
                    </div>
                  </div>
                  {product.tags && (
                    <div style={{ marginTop: 10 }}>
                      {product.tags.split(",").map((tag) => (
                        <Tag key={tag} style={{ marginBottom: 4 }}>
                          {tag}
                        </Tag>
                      ))}
                    </div>
                  )}
                </Card>
              </Col>
            ))}
          </Row>
        )}
      </Spin>

      {total > PAGE_SIZE && (
        <div style={{ textAlign: "center", marginTop: 28 }}>
          <Pagination
            current={page}
            pageSize={PAGE_SIZE}
            total={total}
            showSizeChanger={false}
            onChange={setPage}
          />
        </div>
      )}
    </>
  );
}
