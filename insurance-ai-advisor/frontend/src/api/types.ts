/** 与后端 schemas 对应的类型定义。字段名保持一致，改后端要同步改这里。 */

export interface Page<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  pages: number;
}

export interface UserInfo {
  id: number;
  phone: string;
  nickname: string | null;
  real_name: string | null;
  is_active: boolean;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: UserInfo;
}

export interface Product {
  id: number;
  code: string;
  name: string;
  company: string;
  category: string;
  coverage_amount: string;
  premium_min: string;
  premium_max: string;
  age_min: number;
  age_max: number;
  term_options: string;
  waiting_period_days: number;
  tags: string | null;
  status: string;
}

export interface Clause {
  id: number;
  clause_no: string;
  title: string;
  content: string;
}

export interface ProductDetail extends Product {
  description: string | null;
  clauses: Clause[];
  created_at: string;
}

export interface PremiumCalcResult {
  product_id: number;
  product_name: string;
  insured_age: number;
  coverage_amount: string;
  term_years: number;
  base_premium: string;
  age_factor: string;
  term_factor: string;
  premium: string;
  breakdown: string;
}

export interface Order {
  id: number;
  order_no: string;
  product_id: number;
  product_name: string | null;
  premium: string;
  coverage_amount: string;
  term_years: number;
  insured_name: string;
  insured_age: number;
  status: string;
  status_text: string;
  created_at: string;
  paid_at: string | null;
}

export interface PayResult {
  order: Order;
  policy_no: string;
  policy_status: string;
}

export interface Policy {
  id: number;
  policy_no: string;
  product_id: number;
  product_name: string | null;
  holder_name: string;
  insured_name: string;
  beneficiary: string;
  coverage_amount: string;
  premium: string;
  start_date: string;
  end_date: string;
  status: string;
  status_text: string;
}

export interface Claim {
  id: number;
  claim_no: string;
  policy_no: string | null;
  claim_type: string;
  amount: string;
  approved_amount: string | null;
  description: string | null;
  status: string;
  status_text: string;
  submitted_at: string | null;
}

export interface Citation {
  clause_id: number;
  product_name: string;
  clause_no: string;
  title: string;
  content: string;
}

export interface ToolCallTrace {
  name: string;
  arguments: Record<string, unknown>;
  result_preview: string;
}

export interface ChatDonePayload {
  conversation_id: number;
  message_id: number;
  intent: string;
  citations: Citation[];
  tool_calls: ToolCallTrace[];
  latency_ms: number;
  total_tokens: number | null;
}
