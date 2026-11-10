import { Result, Button, Spin } from "antd";
import type { ReactNode } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { useAuth } from "@/store/auth";

/** 受保护页面的包装：未登录就引导去登录，并记住原来想去的地址。 */
export default function RequireAuth({ children }: { children: ReactNode }) {
  const { user, ready } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  if (!ready) {
    return (
      <div style={{ textAlign: "center", padding: "80px 0" }}>
        <Spin />
      </div>
    );
  }

  if (!user) {
    return (
      <Result
        status="403"
        title="需要登录"
        subTitle="登录后才能查看这份信息"
        extra={
          <Button
            type="primary"
            onClick={() => navigate("/login", { state: { from: location.pathname } })}
          >
            去登录
          </Button>
        }
      />
    );
  }

  return <>{children}</>;
}
