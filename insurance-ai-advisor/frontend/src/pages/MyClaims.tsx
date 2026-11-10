import {
  Alert,
  Button,
  Card,
  Empty,
  Form,
  Input,
  InputNumber,
  Modal,
  Select,
  Table,
  Tag,
  message,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { ApiError } from "@/api/client";
import { claimApi, policyApi } from "@/api/endpoints";
import type { Claim, Policy } from "@/api/types";

const STATUS_COLOR: Record<string, string> = {
  submitted: "orange",
  reviewing: "processing",
  approved: "green",
  rejected: "red",
  paid: "blue",
};

const CLAIM_TYPES = ["门诊", "住院", "重疾", "意外"];
const money = (value: string) => Number(value).toLocaleString("zh-CN");

interface ClaimFormValues {
  policy_no: string;
  claim_type: string;
  amount: number;
  description?: string;
}

export default function MyClaims() {
  const navigate = useNavigate();
  const [items, setItems] = useState<Claim[]>([]);
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [loading, setLoading] = useState(true);
  const [open, setOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [form] = Form.useForm<ClaimFormValues>();

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [claims, policyPage] = await Promise.all([
        claimApi.list({ page_size: 50 }),
        policyApi.list({ status: "active", page_size: 50 }),
      ]);
      setItems(claims.items);
      setPolicies(policyPage.items);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const submit = async (values: ClaimFormValues) => {
    setSubmitting(true);
    try {
      const claim = await claimApi.submit(values);
      message.success(`理赔申请已提交，单号 ${claim.claim_no}`);
      setOpen(false);
      form.resetFields();
      await load();
    } catch (error) {
      message.error(error instanceof ApiError ? error.message : "提交失败");
    } finally {
      setSubmitting(false);
    }
  };

  const columns: ColumnsType<Claim> = [
    { title: "理赔单号", dataIndex: "claim_no", width: 230, ellipsis: true },
    { title: "保单号", dataIndex: "policy_no", width: 230, ellipsis: true },
    { title: "出险类型", dataIndex: "claim_type", width: 100 },
    {
      title: "申请金额",
      dataIndex: "amount",
      width: 130,
      render: (value: string) => `¥${money(value)}`,
    },
    {
      title: "状态",
      dataIndex: "status",
      width: 110,
      render: (status: string, row) => (
        <Tag color={STATUS_COLOR[status] ?? "default"}>{row.status_text}</Tag>
      ),
    },
    {
      title: "提交时间",
      dataIndex: "submitted_at",
      width: 180,
      render: (value: string | null) => (value ? value.slice(0, 19).replace("T", " ") : "-"),
    },
  ];

  return (
    <>
      <h1 className="page-title">理赔服务</h1>
      <div className="page-subtitle">提交后可在下方查看进度，也可以直接问右下角的 AI 顾问</div>

      <Card
        title="理赔记录"
        extra={
          <Button type="primary" onClick={() => setOpen(true)} disabled={policies.length === 0}>
            申请理赔
          </Button>
        }
      >
        {items.length === 0 && !loading ? (
          <Empty
            description={
              policies.length === 0 ? "还没有生效中的保单，暂时无法申请理赔" : "还没有理赔记录"
            }
          >
            {policies.length === 0 && (
              <Button type="primary" onClick={() => navigate("/products")}>
                去投保
              </Button>
            )}
          </Empty>
        ) : (
          <Table<Claim>
            rowKey="id"
            loading={loading}
            columns={columns}
            dataSource={items}
            pagination={{ pageSize: 10, hideOnSinglePage: true }}
            scroll={{ x: 1000 }}
          />
        )}
      </Card>

      <Modal
        title="提交理赔申请"
        open={open}
        onCancel={() => setOpen(false)}
        onOk={() => form.submit()}
        confirmLoading={submitting}
        okText="提交申请"
      >
        <Alert
          type="warning"
          style={{ marginBottom: 16 }}
          message="请确保出险情况属实，虚假理赔将承担法律责任"
        />
        <Form<ClaimFormValues> form={form} layout="vertical" onFinish={submit}>
          <Form.Item
            name="policy_no"
            label="理赔依据保单"
            rules={[{ required: true, message: "请选择保单" }]}
          >
            <Select
              placeholder="选择生效中的保单"
              options={policies.map((policy) => ({
                value: policy.policy_no,
                label: `${policy.product_name ?? ""}（${policy.policy_no}）`,
              }))}
            />
          </Form.Item>
          <Form.Item
            name="claim_type"
            label="出险类型"
            rules={[{ required: true, message: "请选择出险类型" }]}
          >
            <Select options={CLAIM_TYPES.map((type) => ({ value: type, label: type }))} />
          </Form.Item>
          <Form.Item
            name="amount"
            label="申请理赔金额"
            rules={[{ required: true, message: "请输入金额" }]}
          >
            <InputNumber min={1} step={100} style={{ width: "100%" }} addonAfter="元" />
          </Form.Item>
          <Form.Item name="description" label="出险描述（选填）">
            <Input.TextArea rows={3} maxLength={500} showCount placeholder="简要描述出险经过" />
          </Form.Item>
        </Form>
      </Modal>
    </>
  );
}
