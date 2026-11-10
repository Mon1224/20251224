import { Button, Card, Empty, Table, Tag } from "antd";
import type { ColumnsType } from "antd/es/table";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { policyApi } from "@/api/endpoints";
import type { Policy } from "@/api/types";

const STATUS_COLOR: Record<string, string> = {
  active: "green",
  pending: "orange",
  expired: "default",
  claimed: "blue",
  surrendered: "default",
};

const money = (value: string) => Number(value).toLocaleString("zh-CN");

export default function MyPolicies() {
  const navigate = useNavigate();
  const [items, setItems] = useState<Policy[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const result = await policyApi.list({ page_size: 50 });
        setItems(result.items);
      } finally {
        setLoading(false);
      }
    };
    void load();
  }, []);

  const columns: ColumnsType<Policy> = [
    { title: "保单号", dataIndex: "policy_no", width: 230, ellipsis: true },
    { title: "产品", dataIndex: "product_name", ellipsis: true },
    { title: "投保人", dataIndex: "holder_name", width: 100 },
    { title: "被保人", dataIndex: "insured_name", width: 100 },
    {
      title: "保额",
      dataIndex: "coverage_amount",
      width: 130,
      render: (value: string) => `¥${money(value)}`,
    },
    {
      title: "保障期间",
      key: "period",
      width: 210,
      render: (_, row) => `${row.start_date} ~ ${row.end_date}`,
    },
    {
      title: "状态",
      dataIndex: "status",
      width: 100,
      render: (status: string, row) => (
        <Tag color={STATUS_COLOR[status] ?? "default"}>{row.status_text}</Tag>
      ),
    },
    {
      title: "操作",
      key: "action",
      width: 100,
      render: () => (
        <Button size="small" type="link" onClick={() => navigate("/claims")}>
          申请理赔
        </Button>
      ),
    },
  ];

  return (
    <>
      <h1 className="page-title">我的保单</h1>
      <div className="page-subtitle">保单是理赔的凭据，只有生效中的保单才能申请理赔</div>

      <Card>
        {items.length === 0 && !loading ? (
          <Empty description="还没有保单">
            <Button type="primary" onClick={() => navigate("/products")}>
              去投保
            </Button>
          </Empty>
        ) : (
          <Table<Policy>
            rowKey="id"
            loading={loading}
            columns={columns}
            dataSource={items}
            pagination={{ pageSize: 10, hideOnSinglePage: true }}
            scroll={{ x: 1000 }}
          />
        )}
      </Card>
    </>
  );
}
