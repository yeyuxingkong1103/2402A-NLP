"use client";

/**
 * 站点入口：按登录态分流。
 *
 * 放在客户端判断的原因：会话令牌存在 localStorage，服务端渲染时读不到。
 * 若用 middleware 重定向，SSR 阶段拿不到令牌会先渲染登录页再跳转，出现闪烁。
 */

import { useRouter } from "next/navigation";
import { useEffect } from "react";

import { LoadingDots } from "@/components/ui/primitives";
import { readAccessToken } from "@/lib/auth-token";

export default function HomePage() {
  const router = useRouter();

  useEffect(() => {
    router.replace(readAccessToken() ? "/chat" : "/login");
  }, [router]);

  return (
    <div className="flex flex-1 items-center justify-center py-24">
      <LoadingDots label="正在进入…" />
    </div>
  );
}
