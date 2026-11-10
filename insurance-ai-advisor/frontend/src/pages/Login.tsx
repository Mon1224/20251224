import { Button, Card, Form, Input, Typography, message } from "antd";
import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";

import { ApiError } from "@/api/client";
import { useAuth } from "@/store/auth";

interface FormValues {
  phone: string;
  password: string;
}

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [loading, setLoading] = useState(false);

  // 从受保护页面被踢过来时，登录后回到原处
  const from = (location.state as { from?: string } | null)?.from ?? "/products";

  const onFinish = async (values: FormValues) => {
    setLoading(true);
    try {
      await login(values.phone, values.password);
      message.success("登录成功");
      navigate(from, { replace: true });
    } catch (error) {
      message.error(error instanceof ApiError ? error.message : "登录失败，请稍后重试");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#f5f6f8",
      }}
    >
      <Card style={{ width: 380 }}>
        <Typography.Title level={3} style={{ textAlign: "center", color: "#0f766e" }}>
          安心保
        </Typography.Title>
        <Typography.Paragraph type="secondary" style={{ textAlign: "center" }}>
          登录后可使用 AI 保险顾问
        </Typography.Paragraph>

        <Form<FormValues> layout="vertical" onFinish={onFinish} requiredMark={false}>
          <Form.Item
            name="phone"
            label="手机号"
            rules={[
              { required: true, message: "请输入手机号" },
              { pattern: /^1[3-9]\d{9}$/, message: "手机号格式不正确" },
            ]}
          >
            <Input size="large" placeholder="13800000000" autoComplete="username" />
          </Form.Item>
          <Form.Item
            name="password"
            label="密码"
            rules={[{ required: true, message: "请输入密码" }]}
          >
            <Input.Password size="large" placeholder="请输入密码" autoComplete="current-password" />
          </Form.Item>
          <Button type="primary" htmlType="submit" size="large" block loading={loading}>
            登录
          </Button>
        </Form>

        <div style={{ marginTop: 16, textAlign: "center", fontSize: 13 }}>
          还没有账号？<Link to="/register">立即注册</Link>
        </div>
        <div style={{ marginTop: 8, textAlign: "center", fontSize: 12, color: "#9ca3af" }}>
          演示账号：13800000000 / demo1234
        </div>
      </Card>
    </div>
  );
}
