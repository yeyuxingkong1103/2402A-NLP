import type { NextRequest } from "next/server";

/**
 * 运行时反向代理：/api/* -> ${BACKEND_ORIGIN}/api/*
 *
 * 为什么不用 next.config.ts 的 rewrites()：
 * rewrites 的目标地址在 `next build` 时就被写死进 routes-manifest.json，
 * 运行阶段再设置 BACKEND_ORIGIN 不会生效，换后端必须重新构建。
 * 这里在请求到达时读取 process.env.BACKEND_ORIGIN，因此同一份构建产物
 * 可以在开发机 / 测试机 / 生产容器里指向不同的后端。
 *
 * 浏览器只看到同源请求（前端域名 + /api/...），不存在 CORS 问题，
 * 也不会把后端内网地址暴露到浏览器。
 */

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const DEFAULT_BACKEND_ORIGIN = "http://127.0.0.1:8000";

/** 依据请求头判断上游需要的压缩编码，避免把 gzip 实体按明文转发。 */
function resolveAcceptEncoding(req: NextRequest): string | null {
  const raw = req.headers.get("accept-encoding");
  if (!raw) return null;
  return raw;
}

async function proxy(req: NextRequest, ctx: { params: Promise<{ path?: string[] }> }) {
  const backendOrigin = (process.env.BACKEND_ORIGIN ?? DEFAULT_BACKEND_ORIGIN).replace(/\/+$/, "");

  const { path } = await ctx.params;
  const suffix = (path ?? []).map(encodeURIComponent).join("/");
  const target = `${backendOrigin}/api/${suffix}${req.nextUrl.search}`;

  // 逐跳头（hop-by-hop）不能透传，否则会破坏上游连接语义
  const headers = new Headers(req.headers);
  headers.delete("host");
  headers.delete("connection");
  headers.delete("keep-alive");
  headers.delete("transfer-encoding");
  headers.delete("upgrade");
  headers.delete("content-length");
  headers.delete("accept-encoding");

  const method = req.method.toUpperCase();
  const hasBody = !(method === "GET" || method === "HEAD");

  const init: RequestInit & { duplex?: "half" } = {
    method,
    headers,
    redirect: "manual",
    cache: "no-store",
  };

  if (hasBody) {
    // SSE 与长响应需要流式转发，不能先缓冲整个响应体
    init.body = req.body;
    init.duplex = "half";
  }

  let upstream: Response;
  try {
    upstream = await fetch(target, init as RequestInit);
  } catch (err) {
    // 后端未启动 / 地址不可达：给出可读的 502，而不是让浏览器拿到空白页
    const detail = err instanceof Error ? err.message : String(err);
    return new Response(
      JSON.stringify({
        code: 50003,
        message: `无法连接后端服务（${backendOrigin}），请确认后端已启动。`,
        data: null,
        request_id: req.headers.get("x-request-id") ?? "",
        detail,
      }),
      {
        status: 502,
        headers: { "content-type": "application/json; charset=utf-8" },
      },
    );
  }

  // 透传上游响应头，但同样剔除逐跳头，并移除可能冲突的编码声明
  const respHeaders = new Headers();
  upstream.headers.forEach((value, key) => {
    const k = key.toLowerCase();
    if (
      k === "connection" ||
      k === "keep-alive" ||
      k === "transfer-encoding" ||
      k === "upgrade" ||
      k === "content-encoding" ||
      k === "content-length"
    ) {
      return;
    }
    respHeaders.set(key, value);
  });

  // 明确告知中间层与浏览器：SSE 不能被缓冲或改写
  const contentType = upstream.headers.get("content-type") ?? "";
  if (contentType.includes("text/event-stream")) {
    respHeaders.set("cache-control", "no-cache, no-transform");
    respHeaders.set("x-accel-buffering", "no");
  }

  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: respHeaders,
  });
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
export const OPTIONS = proxy;
export const HEAD = proxy;
