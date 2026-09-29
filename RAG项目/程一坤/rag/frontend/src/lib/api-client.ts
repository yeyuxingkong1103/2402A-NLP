/**
 * 后端 HTTP 客户端。
 *
 * 职责：
 * 1. 统一拼 /api/v1 前缀、注入 Authorization 与 Content-Type；
 * 2. 把「HTTP 状态码异常」与「业务 code 非 0」两种失败统一成一个 ApiError，
 *    调用方只需要 catch 一种错误；
 * 3. 遇到 40100 自动清空本地令牌，让界面回到未登录态。
 *
 * 不在这里做 toast：错误提示怎么展示属于界面决策，客户端只负责如实抛出。
 */

import { clearAccessToken, readAccessToken } from "@/lib/auth-token";
import { ErrorCode, describeErrorCode } from "@/lib/error-codes";
import type { ApiEnvelope } from "@/lib/types";

/** API 基础路径，与 docs/接口文档.md 2 节一致 */
export const API_BASE_PATH = "/api/v1";

/** 统一失败类型：HTTP 层与业务层错误都收敛到这里 */
export class ApiError extends Error {
  /** 业务错误码；网络层失败时为 null */
  readonly code: number | null;
  /** HTTP 状态码；网络层失败时为 null */
  readonly httpStatus: number | null;
  /** 后端返回的 request_id，便于对着服务端日志排查 */
  readonly requestId: string | null;

  constructor(params: {
    message: string;
    code: number | null;
    httpStatus: number | null;
    requestId?: string | null;
  }) {
    super(params.message);
    this.name = "ApiError";
    this.code = params.code;
    this.httpStatus = params.httpStatus;
    this.requestId = params.requestId ?? null;
  }

  /** 是否为登录态失效 */
  get isUnauthorized(): boolean {
    return this.code === ErrorCode.UNAUTHORIZED;
  }
}

interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "DELETE";
  body?: unknown;
  /** 是否携带令牌；默认 true */
  auth?: boolean;
  /** 外部取消信号，例如组件卸载或用户点击停止 */
  signal?: AbortSignal;
}

/**
 * 生成一个 request_id 并随请求发出。
 *
 * 后端 RequestIdMiddleware 会优先沿用外部传入的 X-Request-ID，
 * 这样浏览器报错时能直接把同一个 ID 拿去后端日志里搜。
 */
function createRequestId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `web_${crypto.randomUUID().replace(/-/g, "").slice(0, 12)}`;
  }
  return `web_${Math.random().toString(16).slice(2, 14)}`;
}

/**
 * 发起一次普通 JSON 请求并解包 data。
 *
 * 失败一律抛 ApiError；成功时返回 data（后端保证成功响应 data 非 null）。
 */
export async function requestJson<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const { method = "POST", body, auth = true, signal } = options;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "X-Request-ID": createRequestId(),
  };

  if (auth) {
    const token = readAccessToken();
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE_PATH}${path}`, {
      method,
      headers,
      signal,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new ApiError({
      message: "无法连接服务，请检查网络或服务是否已启动",
      code: null,
      httpStatus: null,
    });
  }

  return unwrapResponse<T>(response);
}

/** 解包响应：把 HTTP 错误与业务错误统一成 ApiError */
async function unwrapResponse<T>(response: Response): Promise<T> {
  let payload: ApiEnvelope<T> | null = null;
  try {
    payload = (await response.json()) as ApiEnvelope<T>;
  } catch {
    // 后端异常时可能返回非 JSON（如网关错误页），此时只保留状态码信息
    payload = null;
  }

  const requestId = payload?.request_id ?? response.headers.get("X-Request-ID");

  if (!response.ok) {
    const code = payload?.code ?? null;
    const isUnauthorized =
      response.status === 401 || code === ErrorCode.UNAUTHORIZED;
    const message = isUnauthorized
      ? describeErrorCode(ErrorCode.UNAUTHORIZED, "")
      : describeErrorCode(
          code ?? -1,
          payload?.message ?? `请求失败（HTTP ${response.status}）`,
        );
    if (isUnauthorized) {
      clearAccessToken();
    }
    throw new ApiError({
      message,
      code: isUnauthorized ? ErrorCode.UNAUTHORIZED : code,
      httpStatus: response.status,
      requestId,
    });
  }

  if (!payload) {
    throw new ApiError({
      message: "服务返回了无法解析的内容",
      code: null,
      httpStatus: response.status,
      requestId,
    });
  }

  if (payload.code !== 0) {
    if (payload.code === ErrorCode.UNAUTHORIZED) {
      clearAccessToken();
    }
    throw new ApiError({
      message: describeErrorCode(payload.code, payload.message),
      code: payload.code,
      httpStatus: response.status,
      requestId,
    });
  }

  if (payload.data === null) {
    throw new ApiError({
      message: "服务未返回数据",
      code: payload.code,
      httpStatus: response.status,
      requestId,
    });
  }

  return payload.data;
}

/**
 * 发起 SSE 流式请求，返回原始 Response 交给上层逐行解析。
 *
 * 单独成一个函数的原因：SSE 不能走 requestJson 那套「读完再解包」，
 * 必须把 body 留给调用方流式读取。
 */
export async function requestStream(
  path: string,
  options: RequestOptions = {},
): Promise<Response> {
  const { method = "POST", body, auth = true, signal } = options;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "text/event-stream",
    "X-Request-ID": createRequestId(),
  };

  if (auth) {
    const token = readAccessToken();
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE_PATH}${path}`, {
      method,
      headers,
      signal,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new ApiError({
      message: "无法连接服务，请检查网络或服务是否已启动",
      code: null,
      httpStatus: null,
    });
  }

  if (!response.ok) {
    await unwrapResponse<never>(response);
  }

  return response;
}
