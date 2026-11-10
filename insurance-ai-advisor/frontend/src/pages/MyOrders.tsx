import { Button, Card, Empty, Popconfirm, Space, Table, Tag, message } from "antd";
import type { ColumnsType } from "antd/es/table";
import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { ApiError } from "@/api/client";
import { orderApi } from "@/api/endpoints";
import type { Order } from "@/api/types";

const STATUS_COLOR: Record<string, string> = {
  pending: "orange",
  paid: "green",
  cancelled: "default",
};

const money = (value: string) => Number(value).toLocaleString("zh-CN");

export default function MyOrders() {
  const navigate = useNavigate();
  const [items, setItems] = useState<Order[]>([]);
  const [loading, setLoading] = useState(true);
  const [acting, setActing] = useState<string>();

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const result = await orderApi.list({ page_size: 50 });
      setItems(result.items);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const pay = async (orderNo: string) => {
    setActing(orderNo);
    try {
      const result = await orderApi.pay(orderNo);
      message.success(`支付成功，已生成保单 ${result.policy_no}`);
      await load();
    } catch (error) {
      message.error(error instanceof ApiError ? error.message : "支付失败");
    } finally {
      setActing(undefined);
    }
  };

  const cancel = async (orderNo: string) => {
    setActing(orderNo);
    try {
      await orderApi.cancel(orderNo);
      message.success("订单已取消");
      await load();
    } catch (error) {
      message.error(error instanceof ApiError ? error.message : "取消失败");
    } finally {
      setActing(undefined);
    }
  };

  const columns: ColumnsType<Order> = [
    { title: "订单号", dataIndex: "order_no", width: 230, ellipsis: true },
    { title: "产品", dataIndex: "product_name", ellipsis: true },
    {
      title: "保费",
      dataIndex: "premium",
      width: 120,
      render: (value: string) => `¥${money(value)}`,
    },
    { title: "被保人", dataIndex: "insured_name", width: 100 },
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
      width: 170,
      render: (_, row) => (
        <Space>
          {row.status === "pending" && (
            <>
              <Button
                type="primary"
                size="small"
                loading={acting === row.order_no}
                onClick={() => void pay(row.order_no)}
              >
                支付
              </Button>
              <Popconfirm
                title="确定取消这个订单？"
                onConfirm={() => void cancel(row.order_no)}
                okText="确定"
                cancelText="再想想"
              >
                <Button size="small">取消</Button>
              </Popconfirm>
            </>
          )}
          {row.status === "paid" && (
            <Button size="small" type="link" onClick={() => navigate("/policies")}>
              查看保单
            </Button>
          )}
        </Space>
      ),
    },
  ];

  return (
    <>
      <h1 className="page-title">我的订单</h1>
      <div className="page-subtitle">支付成功后会自动生成保单</div>

      <Card>
        {items.length === 0 && !loading ? (
          <Empty description="还没有订单">
            <Button type="primary" onClick={() => navigate("/products")}>
              去商城看看
            </Button>
          </Empty>
        ) : (
          <Table<Order>
            rowKey="id"
            loading={loading}
            columns={columns}
            dataSource={items}
            pagination={{ pageSize: 10, hideOnSinglePage: true }}
            scroll={{ x: 900 }}
          />
        )}
      </Card>
    </>
  );
}
