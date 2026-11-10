/** 统一的请求封装：拼地址、带令牌、翻译错误、处理登录失效。 */

const API_BASE = import.meta.env.VITE_API_BASE ?? "";
const TOKEN_KEY = "insurance_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string | null): void {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

export class ApiError extends Error {
  code: string;
  status: number;

  constructor(message: string, code: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

/** 后端返回统一格式：{ code, message, detail } */
interface ErrorBody {
  code?: string;
  message?: string;
  detail?: unknown;
}

export async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const response = await fetch(`${API_BASE}${path}`, { ...init, headers });
  const text = await response.text();
  const body = text ? (JSON.parse(text) as unknown) : null;

  if (!response.ok) {
    const error = (body ?? {}) as ErrorBody;
    // 令牌过期：清掉本地令牌并通知外层跳登录页。
    // 用自定义事件而不是直接改 location，避免在请求层耦合路由。
    if (response.status === 401) {
      setToken(null);
      window.dispatchEvent(new CustomEvent("auth:expired"));
    }
    throw new ApiError(
      error.message ?? `请求失败（${response.status}）`,
      error.code ?? "HTTP_ERROR",
      response.status,
    );
  }

  return body as T;
}

export function get<T>(path: string): Promise<T> {
  return request<T>(path);
}

export function post<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

/** 拼查询串，自动跳过 undefined / null / 空字符串 */
export function query(params: Record<string, string | number | undefined | null>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue;
    search.set(key, String(value));
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}
