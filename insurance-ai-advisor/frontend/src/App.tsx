import { Navigate, Route, Routes } from "react-router-dom";

import AppLayout from "@/components/AppLayout";
import RequireAuth from "@/components/RequireAuth";
import Login from "@/pages/Login";
import MyClaims from "@/pages/MyClaims";
import MyOrders from "@/pages/MyOrders";
import MyPolicies from "@/pages/MyPolicies";
import ProductDetail from "@/pages/ProductDetail";
import ProductList from "@/pages/ProductList";
import Register from "@/pages/Register";

export default function App() {
  return (
    <Routes>
      {/* 登录注册是独立页面，不套主框架的导航栏 */}
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />

      <Route element={<AppLayout />}>
        <Route path="/" element={<Navigate to="/products" replace />} />
        <Route path="/products" element={<ProductList />} />
        <Route path="/products/:id" element={<ProductDetail />} />
        <Route
          path="/orders"
          element={
            <RequireAuth>
              <MyOrders />
            </RequireAuth>
          }
        />
        <Route
          path="/policies"
          element={
            <RequireAuth>
              <MyPolicies />
            </RequireAuth>
          }
        />
        <Route
          path="/claims"
          element={
            <RequireAuth>
              <MyClaims />
            </RequireAuth>
          }
        />
      </Route>

      <Route path="*" element={<Navigate to="/products" replace />} />
    </Routes>
  );
}
