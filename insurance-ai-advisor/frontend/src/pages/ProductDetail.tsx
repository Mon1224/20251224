import {
  Alert,
  Button,
  Card,
  Col,
  Collapse,
  Descriptions,
  Form,
  Input,
  InputNumber,
  Modal,
  Row,
  Select,
  Space,
  Spin,
  Tag,
  message,
} from "antd";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { ApiError } from "@/api/client";
import { orderApi, productApi } from "@/api/endpoints";
import type { PremiumCalcResult, ProductDetail as ProductDetailType } from "@/api/types";
import { useAuth } from "@/store/auth";

const money = (value: string | number) => Number(value).toLocaleString("zh-CN");

interface CalcValues {
  insured_age: number;
  coverage_amount: number;
  term_years: number;
}

interface OrderValues extends CalcValues {
  insured_name: string;
  insured_id_card: string;
}

export default function ProductDetail() {
  const { id } = useParams();
  const productId = Number(id);
  const navigate = useNavigate();
  const { user } = useAuth();

  const [product, setProduct] = useState<ProductDetailType | null>(null);
  const [loading, setLoading] = useState(true);
  const [calcResult, setCalcResult] = useState<PremiumCalcResult | null>(null);
  const [calculating, setCalculating] = useState(false);
  const [orderOpen, setOrderOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const [calcForm] = Form.useForm<CalcValues>();
  const [orderForm] = Form.useForm<OrderValues>();

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const detail = await productApi.detail(productId);
        setProduct(detail);
        const terms = detail.term_options
          .split(",")
          .map((item) => Number(item.trim()))
          .filter((item) => Number.isFinite(item));
        calcForm.setFieldsValue({
          insured_age: 30,
          coverage_amount: Number(detail.coverage_amount),
          term_years: terms[0] ?? 1,
        });
      } catch (error) {
        message.error(error instanceof ApiError ? error.message : "加载产品失败");
      } finally {
        setLoading(false);
      }
    };
    void load();
  }, [productId, calcForm]);

  const termOptions = useMemo(() => {
    if (!product) return [];
    return product.term_options
      .split(",")
      .map((item) => Number(item.trim()))
      .filter((item) => Number.isFinite(item))
      .map((year) => ({ value: year, label: `${year} 年` }));
  }, [product]);

  const onCalc = useCallback(
    async (values: CalcValues) => {
      setCalculating(true);
      try {
        const result = await productApi.calcPremium({ product_id: productId, ...values });
        setCalcResult(result);
      } catch (error) {
        setCalcResult(null);
        message.warning(error instanceof ApiError ? error.message : "试算失败");
      } finally {
        setCalculating(false);
      }
    },
    [productId],
  );

  const openOrder = () => {
    if (!user) {
      message.info("请先登录再投保");
      navigate("/login", { state: { from: `/products/${productId}` } });
      return;
    }
    const values = calcForm.getFieldsValue();
    orderForm.setFieldsValue({
      ...values,
      insured_name: user.real_name ?? "",
    });
    setOrderOpen(true);
  };

  const submitOrder = async (values: OrderValues) => {
    setSubmitting(true);
    try {
      const order = await orderApi.create({ product_id: productId, ...values });
      message.success(`下单成功，订单号 ${order.order_no}`);
      setOrderOpen(false);
      navigate("/orders");
    } catch (error) {
      message.error(error instanceof ApiError ? error.message : "下单失败");
    } finally {
      setSubmitting(false);
    }
  };

  if (loading || !product) {
    return (
      <div style={{ textAlign: "center", padding: "80px 0" }}>
        <Spin />
      </div>
    );
  }

  return (
    <>
      <Button type="link" style={{ paddingLeft: 0 }} onClick={() => navigate("/products")}>
        ← 返回商城
      </Button>

      <h1 className="page-title">{product.name}</h1>
      <div className="page-subtitle">
        {product.company} · {product.category}
      </div>

      <Row gutter={20}>
        <Col xs={24} lg={14}>
          <Card title="产品信息" style={{ marginBottom: 20 }}>
            <Descriptions column={2} size="small">
              <Descriptions.Item label="参考保额">
                {money(product.coverage_amount)} 元
              </Descriptions.Item>
              <Descriptions.Item label="年缴保费">
                {money(product.premium_min)} ~ {money(product.premium_max)} 元
              </Descriptions.Item>
              <Descriptions.Item label="投保年龄">
                {product.age_min}-{product.age_max} 岁
              </Descriptions.Item>
              <Descriptions.Item label="可选期间">{product.term_options} 年</Descriptions.Item>
              <Descriptions.Item label="等待期">
                {product.waiting_period_days === 0 ? "无等待期" : `${product.waiting_period_days} 天`}
              </Descriptions.Item>
              <Descriptions.Item label="产品编码">{product.code}</Descriptions.Item>
            </Descriptions>
            {product.description && (
              <div style={{ marginTop: 12, color: "#4b5563", lineHeight: 1.9 }}>
                {product.description}
              </div>
            )}
            <div style={{ marginTop: 12 }}>
              {(product.tags ?? "").split(",").filter(Boolean).map((tag) => (
                <Tag key={tag}>{tag}</Tag>
              ))}
            </div>
          </Card>

          <Card title={`保险条款（共 ${product.clauses.length} 条）`}>
            <Collapse
              items={product.clauses.map((clause) => ({
                key: String(clause.id),
                label: `第 ${clause.clause_no} 条　${clause.title}`,
                children: (
                  <div style={{ lineHeight: 2, color: "#4b5563", whiteSpace: "pre-wrap" }}>
                    {clause.content}
                  </div>
                ),
              }))}
            />
          </Card>
        </Col>

        <Col xs={24} lg={10}>
          <Card title="保费试算" style={{ position: "sticky", top: 80 }}>
            <Form<CalcValues> form={calcForm} layout="vertical" onFinish={onCalc}>
              <Form.Item
                name="insured_age"
                label="被保人年龄"
                rules={[{ required: true, message: "请输入年龄" }]}
              >
                <InputNumber min={0} max={100} style={{ width: "100%" }} addonAfter="岁" />
              </Form.Item>
              <Form.Item
                name="coverage_amount"
                label="投保保额"
                rules={[{ required: true, message: "请输入保额" }]}
              >
                <InputNumber<number>
                  min={10000}
                  step={10000}
                  style={{ width: "100%" }}
                  addonAfter="元"
                  formatter={(value) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
                  parser={(value) => Number((value ?? "").replace(/,/g, ""))}
                />
              </Form.Item>
              <Form.Item
                name="term_years"
                label="保险期间"
                rules={[{ required: true, message: "请选择期间" }]}
              >
                <Select options={termOptions} />
              </Form.Item>

              <Space style={{ width: "100%" }}>
                <Button htmlType="submit" loading={calculating}>
                  试算保费
                </Button>
                <Button type="primary" onClick={openOrder}>
                  立即投保
                </Button>
              </Space>
            </Form>

            {calcResult && (
              <Alert
                type="success"
                style={{ marginTop: 18 }}
                message={
                  <span>
                    预计保费 <b>{money(calcResult.premium)} 元 / 年</b>
                  </span>
                }
                description={
                  <div style={{ fontSize: 12.5, lineHeight: 1.9, color: "#4b5563" }}>
                    {calcResult.breakdown}
                  </div>
                }
              />
            )}
          </Card>
        </Col>
      </Row>

      <Modal
        title="填写投保信息"
        open={orderOpen}
        onCancel={() => setOrderOpen(false)}
        onOk={() => orderForm.submit()}
        confirmLoading={submitting}
        okText="提交投保"
      >
        <Alert
          type="info"
          style={{ marginBottom: 16 }}
          message="保费由系统按条款费率计算，以最终核保结果为准"
        />
        <Form<OrderValues> form={orderForm} layout="vertical" onFinish={submitOrder}>
          <Form.Item
            name="insured_name"
            label="被保人姓名"
            rules={[{ required: true, message: "请输入被保人姓名" }]}
          >
            <Input placeholder="与身份证一致" />
          </Form.Item>
          <Form.Item
            name="insured_id_card"
            label="被保人身份证号"
            rules={[
              { required: true, message: "请输入身份证号" },
              { min: 15, max: 18, message: "身份证号长度不正确" },
            ]}
          >
            <Input placeholder="18 位身份证号" />
          </Form.Item>
          <Form.Item
            name="insured_age"
            label="被保人年龄"
            rules={[{ required: true, message: "请输入年龄" }]}
          >
            <InputNumber
              min={product.age_min}
              max={product.age_max}
              style={{ width: "100%" }}
              addonAfter="岁"
            />
          </Form.Item>
          <Form.Item
            name="coverage_amount"
            label="投保保额"
            rules={[{ required: true, message: "请输入保额" }]}
          >
            <InputNumber min={10000} step={10000} style={{ width: "100%" }} addonAfter="元" />
          </Form.Item>
          <Form.Item
            name="term_years"
            label="保险期间"
            rules={[{ required: true, message: "请选择期间" }]}
          >
            <Select options={termOptions} />
          </Form.Item>
        </Form>
      </Modal>
    </>
  );
}
