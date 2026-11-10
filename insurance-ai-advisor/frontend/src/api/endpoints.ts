/** 后端接口的调用入口。页面里不出现裸路径，全部从这里走。 */

import { get, post, query } from "@/api/client";
import type {
  Claim,
  Order,
  Page,
  PayResult,
  Policy,
  PremiumCalcResult,
  Product,
  ProductDetail,
  TokenResponse,
  UserInfo,
} from "@/api/types";

// ---------------------------------------------------------------- 用户
export const authApi = {
  register: (body: { phone: string; password: string; nickname?: string }) =>
    post<TokenResponse>("/api/v1/auth/register", body),
  login: (body: { phone: string; password: string }) =>
    post<TokenResponse>("/api/v1/auth/login", body),
  me: () => get<UserInfo>("/api/v1/auth/me"),
  logout: () => post<{ message: string }>("/api/v1/auth/logout"),
};

// ---------------------------------------------------------------- 产品
export const productApi = {
  list: (params: { category?: string; keyword?: string; page?: number; page_size?: number }) =>
    get<Page<Product>>(`/api/v1/products${query(params)}`),
  detail: (id: number) => get<ProductDetail>(`/api/v1/products/${id}`),
  calcPremium: (body: {
    product_id: number;
    insured_age: number;
    coverage_amount: number;
    term_years: number;
  }) => post<PremiumCalcResult>("/api/v1/products/premium", body),
};

// ---------------------------------------------------------------- 订单
export const orderApi = {
  create: (body: {
    product_id: number;
    coverage_amount: number;
    term_years: number;
    insured_name: string;
    insured_id_card: string;
    insured_age: number;
  }) => post<Order>("/api/v1/orders", body),
  list: (params: { status?: string; page?: number; page_size?: number } = {}) =>
    get<Page<Order>>(`/api/v1/orders${query(params)}`),
  pay: (orderNo: string) => post<PayResult>(`/api/v1/orders/${orderNo}/pay`),
  cancel: (orderNo: string) => post<Order>(`/api/v1/orders/${orderNo}/cancel`),
};

// ---------------------------------------------------------------- 保单
export const policyApi = {
  list: (params: { status?: string; page?: number; page_size?: number } = {}) =>
    get<Page<Policy>>(`/api/v1/policies${query(params)}`),
  detail: (policyNo: string) => get<Policy>(`/api/v1/policies/${policyNo}`),
};

// ---------------------------------------------------------------- 理赔
export const claimApi = {
  submit: (body: { policy_no: string; claim_type: string; amount: number; description?: string }) =>
    post<Claim>("/api/v1/claims", body),
  list: (params: { status?: string; page?: number; page_size?: number } = {}) =>
    get<Page<Claim>>(`/api/v1/claims${query(params)}`),
};
