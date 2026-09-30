"use client";

/**
 * 顶部导航。
 *
 * 承担两件事：
 * 1. 在「问答」与「检索调试」之间切换；
 * 2. 展示登录态并提供注销入口。
 *
 * 登录态直接订阅 localStorage 的令牌变更事件，不引入全局状态库 ——
 * 全局只有一个字符串需要同步，上 Redux/Zustand 属于过度设计。
 */

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { Button, cx } from "@/components/ui/primitives";
import { fetchCurrentUser, logout } from "@/lib/api-auth";
import { clearAccessToken, readAccessToken, subscribeAccessToken } from "@/lib/auth-token";

const NAV_ITEMS = [
  { href: "/chat", label: "问答" },
  { href: "/search", label: "检索调试" },
] as const;

export function AppHeader() {
  const pathname = usePathname();
  const router = useRouter();
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isLoggingOut, setIsLoggingOut] = useState(false);
  // is_admin 由 GET /users/me 提供（接口文档 3.8）；
  // null 表示「还没查到」，此时不渲染管理端入口，避免闪烁
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);

  // 首次挂载时读一次令牌，之后靠订阅同步
  useEffect(() => {
    let cancelled = false;
    const sync = () => {
      const hasToken = readAccessToken() !== null;
      setIsLoggedIn(hasToken);
      if (!hasToken) {
        setIsAdmin(null);
        return;
      }
      fetchCurrentUser()
        .then((me) => {
          if (!cancelled) setIsAdmin(me.is_admin);
        })
        .catch(() => {
          // 查询失败（含令牌失效被 api-client 清掉）一律不显示管理端入口；
          // 401 清令牌后订阅回调会把 isLoggedIn 同步为 false
          if (!cancelled) setIsAdmin(null);
        });
    };
    sync();
    const unsubscribe = subscribeAccessToken(sync);
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, []);

  const handleLogout = useCallback(async () => {
    setIsLoggingOut(true);
    try {
      // 注销失败也必须清本地令牌，否则界面会卡在「已登录但请求全 401」的状态
      await logout();
    } catch {
      // 失败原因不展示：注销是「结果不重要」的操作，本地清干净即可
    } finally {
      clearAccessToken();
      setIsLoggingOut(false);
      router.push("/login");
    }
  }, [router]);

  return (
    <header className="rule-line sticky top-0 z-20 border-b bg-white/90 backdrop-blur-sm">
      <div className="mx-auto flex h-14 max-w-shell items-center gap-6 px-6">
        <Link href={isLoggedIn ? "/chat" : "/login"} className="flex items-baseline gap-2">
          <span className="font-serif text-lg font-medium tracking-wide text-ink-900">
            法律知识助手
          </span>
          <span className="hidden text-xs text-ink-400 sm:inline">
            中国大陆劳动法
          </span>
        </Link>

        {isLoggedIn ? (
          <nav className="flex items-center gap-1">
            {NAV_ITEMS.map((item) => {
              const isActive = pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cx(
                    "rounded px-3 py-1.5 text-sm transition-colors duration-150",
                    isActive
                      ? "bg-ink-100 font-medium text-ink-900"
                      : "text-ink-600 hover:bg-ink-50 hover:text-ink-900",
                  )}
                >
                  {item.label}
                </Link>
              );
            })}
            {isAdmin ? (
              <Link
                href="/admin/documents"
                className={cx(
                  "rounded px-3 py-1.5 text-sm transition-colors duration-150",
                  pathname.startsWith("/admin")
                    ? "bg-ink-100 font-medium text-ink-900"
                    : "text-ink-600 hover:bg-ink-50 hover:text-ink-900",
                )}
              >
                审核
              </Link>
            ) : null}
          </nav>
        ) : null}

        <div className="ml-auto flex items-center gap-2">
          {isLoggedIn ? (
            <Button
              variant="ghost"
              onClick={handleLogout}
              disabled={isLoggingOut}
              className="px-3 py-1.5 text-xs"
            >
              {isLoggingOut ? "注销中…" : "注销"}
            </Button>
          ) : (
            <>
              <Link
                href="/login"
                className="rounded px-3 py-1.5 text-sm text-ink-600 transition-colors duration-150 hover:bg-ink-50 hover:text-ink-900"
              >
                登录
              </Link>
              <Link
                href="/register"
                className="rounded bg-seal-600 px-3 py-1.5 text-sm text-white transition-colors duration-150 hover:bg-seal-700"
              >
                注册
              </Link>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
