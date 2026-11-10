import { LogoutOutlined, ShoppingCartOutlined, UserOutlined } from "@ant-design/icons";
import { Dropdown, Layout, Menu, Space } from "antd";
import { Link, Outlet, useLocation, useNavigate } from "react-router-dom";

import ChatWidget from "@/components/ChatWidget";
import { useAuth } from "@/store/auth";

const MENU_ITEMS = [
  { key: "/products", label: <Link to="/products">保险商城</Link> },
  { key: "/orders", label: <Link to="/orders">我的订单</Link> },
  { key: "/policies", label: <Link to="/policies">我的保单</Link> },
  { key: "/claims", label: <Link to="/claims">理赔服务</Link> },
];

export default function AppLayout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  // 高亮当前栏目：/products/3 这种子路径也要点亮"保险商城"
  const selected = MENU_ITEMS.map((item) => item.key).find((key) =>
    location.pathname.startsWith(key),
  );

  return (
    <Layout style={{ minHeight: "100vh", background: "#f5f6f8" }}>
      <div className="app-header">
        <Link to="/products" className="app-brand" style={{ textDecoration: "none" }}>
          安心保
        </Link>

        <Menu
          mode="horizontal"
          selectedKeys={selected ? [selected] : []}
          items={MENU_ITEMS}
          style={{ flex: 1, borderBottom: "none", background: "transparent" }}
        />

        {user ? (
          <Dropdown
            menu={{
              items: [
                {
                  key: "orders",
                  icon: <ShoppingCartOutlined />,
                  label: "我的订单",
                  onClick: () => navigate("/orders"),
                },
                { type: "divider" },
                {
                  key: "logout",
                  icon: <LogoutOutlined />,
                  label: "退出登录",
                  onClick: () => void logout().then(() => navigate("/login")),
                },
              ],
            }}
          >
            <Space style={{ cursor: "pointer", whiteSpace: "nowrap" }}>
              <UserOutlined />
              {user.nickname ?? user.phone}
            </Space>
          </Dropdown>
        ) : (
          <Space>
            <Link to="/login">登录</Link>
            <Link to="/register">注册</Link>
          </Space>
        )}
      </div>

      <div className="app-content">
        <Outlet />
      </div>

      {/* 悬浮在右下角，任何页面都能唤起 */}
      <ChatWidget />
    </Layout>
  );
}
