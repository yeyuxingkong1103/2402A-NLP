import type { Metadata, Viewport } from "next";

import { AppHeader } from "@/components/app-header";
import { RISK_NOTICE } from "@/lib/types";
import "@/styles/globals.css";

/**
 * 根布局。
 *
 * 全局常驻免责声明：需求文档 7.2 要求聊天页「免责声明常驻」，
 * 这里把它放在布局层，任何页面都不会漏掉。
 */
export const metadata: Metadata = {
  title: "法律知识助手",
  description:
    "基于 RAG 的中国大陆劳动法法律信息检索与问答系统。内容仅供法律信息参考，不能替代律师出具的正式法律意见。",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body className="min-h-screen font-sans antialiased">
        <div className="flex min-h-screen flex-col">
          <AppHeader />
          <main className="flex flex-1 flex-col">{children}</main>
          <footer className="rule-line border-t px-6 py-3">
            <p className="mx-auto max-w-shell text-center text-xs leading-relaxed text-ink-500">
              {RISK_NOTICE}
            </p>
          </footer>
        </div>
      </body>
    </html>
  );
}
