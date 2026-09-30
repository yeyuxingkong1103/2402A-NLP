"use client";

/**
 * 登录守卫：包住需要登录的页面。
 *
 * 实现方式说明：
 * 后端以 Redis 会话令牌鉴权，令牌在前端 localStorage，
 * 服务端组件读不到，所以必须在客户端做门禁。
 *
 * 这里刻意不引入「全局 AuthContext」——只有「有没有令牌」一个布尔值，
 * 用一次读取 + 订阅令牌变更事件即可，上全局状态库是过度设计。
 */

import { useRouter } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";

import { LoadingDots } from "@/components/ui/primitives";
import { readAccessToken, subscribeAccessToken } from "@/lib/auth-token";

export function RequireAuth({ children }: { children: ReactNode }) {
  const router = useRouter();
  // null 表示「还没检查完」；提前渲染业务界面会闪一下未登录内容
  const [hasToken, setHasToken] = useState<boolean | null>(null);

  useEffect(() => {
    const sync = () => {
      const token = readAccessToken();
      setHasToken(token !== null);
      if (token === null) {
        router.replace("/login");
      }
    };
    sync();
    // 订阅令牌变更：令牌失效（如后端返回 40100）时立即踢回登录页
    return subscribeAccessToken(sync);
  }, [router]);

  if (hasToken !== true) {
    return (
      <div className="flex flex-1 items-center justify-center py-24">
        <LoadingDots label="正在校验登录状态" />
      </div>
    );
  }

  return <>{children}</>;
}
