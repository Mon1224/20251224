import { Button, Card, Form, Input, Typography, message } from "antd";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { ApiError } from "@/api/client";
import { useAuth } from "@/store/auth";

interface FormValues {
  phone: string;
  password: string;
  confirm: string;
  nickname?: string;
}

export default function Register() {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);

  const onFinish = async (values: FormValues) => {
    setLoading(true);
    try {
      await register(values.phone, values.password, values.nickname);
      message.success("注册成功，已自动登录");
      navigate("/products", { replace: true });
    } catch (error) {
      message.error(error instanceof ApiError ? error.message : "注册失败，请稍后重试");
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
          注册安心保
        </Typography.Title>

        <Form<FormValues> layout="vertical" onFinish={onFinish} requiredMark={false}>
          <Form.Item
            name="phone"
            label="手机号"
            rules={[
              { required: true, message: "请输入手机号" },
              { pattern: /^1[3-9]\d{9}$/, message: "手机号格式不正确" },
            ]}
          >
            <Input size="large" placeholder="请输入手机号" autoComplete="username" />
          </Form.Item>
          <Form.Item
            name="password"
            label="密码"
            rules={[
              { required: true, message: "请输入密码" },
              { min: 8, message: "密码至少 8 位" },
              { max: 64, message: "密码最多 64 位" },
            ]}
          >
            <Input.Password size="large" placeholder="8-64 位" autoComplete="new-password" />
          </Form.Item>
          <Form.Item
            name="confirm"
            label="确认密码"
            dependencies={["password"]}
            rules={[
              { required: true, message: "请再次输入密码" },
              ({ getFieldValue }) => ({
                validator(_rule, value) {
                  if (!value || getFieldValue("password") === value) return Promise.resolve();
                  return Promise.reject(new Error("两次输入的密码不一致"));
                },
              }),
            ]}
          >
            <Input.Password size="large" placeholder="再次输入密码" autoComplete="new-password" />
          </Form.Item>
          <Form.Item name="nickname" label="昵称（选填）">
            <Input size="large" placeholder="怎么称呼你" />
          </Form.Item>
          <Button type="primary" htmlType="submit" size="large" block loading={loading}>
            注册
          </Button>
        </Form>

        <div style={{ marginTop: 16, textAlign: "center", fontSize: 13 }}>
          已有账号？<Link to="/login">去登录</Link>
        </div>
      </Card>
    </div>
  );
}
