/** 登录状态。用 React Context，不引入额外的状态管理库——这个规模用不上。 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { ReactNode } from "react";

import { getToken, setToken } from "@/api/client";
import { authApi } from "@/api/endpoints";
import type { UserInfo } from "@/api/types";

interface AuthContextValue {
  user: UserInfo | null;
  ready: boolean;
  login: (phone: string, password: string) => Promise<void>;
  register: (phone: string, password: string, nickname?: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserInfo | null>(null);
  const [ready, setReady] = useState(false);

  // 刷新页面后拿本地令牌去换用户信息，实现"保持登录"
  useEffect(() => {
    let alive = true;

    const bootstrap = async () => {
      if (!getToken()) {
        setReady(true);
        return;
      }
      try {
        const me = await authApi.me();
        if (alive) setUser(me);
      } catch {
        // 令牌失效或后端没起来，都当作未登录处理，不打扰用户
        setToken(null);
      } finally {
        if (alive) setReady(true);
      }
    };

    void bootstrap();
    return () => {
      alive = false;
    };
  }, []);

  // 请求层遇到 401 会派发这个事件，这里统一清空登录态
  useEffect(() => {
    const onExpired = () => setUser(null);
    window.addEventListener("auth:expired", onExpired);
    return () => window.removeEventListener("auth:expired", onExpired);
  }, []);

  const login = useCallback(async (phone: string, password: string) => {
    const result = await authApi.login({ phone, password });
    setToken(result.access_token);
    setUser(result.user);
  }, []);

  const register = useCallback(
    async (phone: string, password: string, nickname?: string) => {
      const result = await authApi.register({ phone, password, nickname });
      setToken(result.access_token);
      setUser(result.user);
    },
    [],
  );

  const logout = useCallback(async () => {
    try {
      await authApi.logout();
    } catch {
      // 退出失败也要清本地状态，否则用户会卡在"已登录"假象里
    }
    setToken(null);
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({ user, ready, login, register, logout }),
    [user, ready, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth 必须在 AuthProvider 内部使用");
  return context;
}
