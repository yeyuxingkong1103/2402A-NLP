/**
 * 认证接口（app/app/auth/router.py）。
 *
 * 路径全部带 /api/v1 前缀，与后端实际注册的 prefix="/api/v1/auth" 一致。
 * 后端要求邮箱只能是 qq.com / foxmail.com，前端做前置校验只是为了少跑一趟网络，
 * 真正的校验仍在后端（app/auth/schemas.py validate_email_domain）。
 */

import { requestJson } from "@/lib/api-client";
import type { AuthTokenData, CurrentUserData, StatusData } from "@/lib/types";

/** 允许注册的邮箱域名，与后端 ALLOWED_DOMAINS 保持一致 */
export const ALLOWED_EMAIL_DOMAINS = ["qq.com", "foxmail.com"] as const;

/** 注册 / 登录成功后的令牌数据 */
export type AuthResult = AuthTokenData;

/** 发送注册验证码 */
export function sendRegisterCode(email: string): Promise<StatusData> {
  return requestJson<StatusData>("/auth/register/code", {
    body: { email },
    auth: false,
  });
}

/** 发送密码重置验证码 */
export function sendPasswordResetCode(email: string): Promise<StatusData> {
  return requestJson<StatusData>("/auth/password-reset/code", {
    body: { email },
    auth: false,
  });
}

/** 注册：邮箱 + 验证码 + 密码 */
export function register(
  email: string,
  code: string,
  password: string,
): Promise<AuthResult> {
  return requestJson<AuthResult>("/auth/register", {
    body: { email, code, password },
    auth: false,
  });
}

/** 登录：邮箱 + 密码 */
export function login(email: string, password: string): Promise<AuthResult> {
  return requestJson<AuthResult>("/auth/login", {
    body: { email, password },
    auth: false,
  });
}

/** 重置密码：邮箱 + 验证码 + 新密码 */
export function resetPassword(
  email: string,
  code: string,
  password: string,
): Promise<StatusData> {
  return requestJson<StatusData>("/auth/password-reset", {
    body: { email, code, password },
    auth: false,
  });
}

/** 注销当前会话；后端对重复注销幂等处理 */
export function logout(): Promise<StatusData> {
  return requestJson<StatusData>("/auth/logout", { body: {} });
}

/**
 * 查询当前登录用户（docs/接口文档.md 3.8）。
 *
 * 管理端入口显隐与记忆开关初始状态都以此为准；
 * 未登录（无令牌）时由调用方自行短路，这里只负责发起请求。
 */
export function fetchCurrentUser(): Promise<CurrentUserData> {
  return requestJson<CurrentUserData>("/users/me", { method: "GET" });
}

/** 邮箱格式与域名是否满足后端要求 */
export function validateEmailDomain(email: string): string | null {
  const normalized = email.trim().toLowerCase();
  if (!normalized) {
    return "请输入邮箱";
  }
  const atIndex = normalized.lastIndexOf("@");
  if (atIndex <= 0 || atIndex === normalized.length - 1) {
    return "邮箱格式不正确";
  }
  const domain = normalized.slice(atIndex + 1);
  if (!(ALLOWED_EMAIL_DOMAINS as readonly string[]).includes(domain)) {
    return "仅支持 QQ 邮箱或 Foxmail 邮箱";
  }
  return null;
}

/** 密码强度校验：与后端 Field(min_length=8, max_length=128) 对齐 */
export function validatePassword(password: string): string | null {
  if (!password) {
    return "请输入密码";
  }
  if (password.length < 8) {
    return "密码至少 8 位";
  }
  if (password.length > 128) {
    return "密码不能超过 128 位";
  }
  return null;
}

/** 6 位数字验证码校验：与后端 pattern="^\\d{6}$" 对齐 */
export function validateVerificationCode(code: string): string | null {
  if (!/^\d{6}$/.test(code.trim())) {
    return "请输入 6 位数字验证码";
  }
  return null;
}
