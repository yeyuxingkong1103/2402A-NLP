"use client";

/**
 * 重置密码页：邮箱验证码 + 新密码。
 *
 * 后端契约（app/auth/router.py）：
 *   1. POST /auth/password-reset/code  { email }           → 发验证码
 *   2. POST /auth/password-reset       { email, code, password } → 重置成功
 *
 * 重置成功后后端会撤销该用户所有旧会话（service.reset_password 里
 * revoke_all_sessions），因此这里不直接发令牌，而是引导回登录页重新登录。
 */

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { AuthShell } from "@/components/auth-shell";
import { Button, Field, Notice } from "@/components/ui/primitives";
import { ApiError } from "@/lib/api-client";
import {
  resetPassword,
  sendPasswordResetCode,
  validateEmailDomain,
  validatePassword,
  validateVerificationCode,
} from "@/lib/api-auth";

const RESEND_COOLDOWN_SECONDS = 60;

export default function PasswordResetPage() {
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [password, setPassword] = useState("");

  const [emailError, setEmailError] = useState<string | null>(null);
  const [codeError, setCodeError] = useState<string | null>(null);
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const [formError, setFormError] = useState<string | null>(null);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSendingCode, setIsSendingCode] = useState(false);
  const [notice, setNotice] = useState<{ tone: "success" | "error"; text: string } | null>(
    null,
  );
  const [countdown, setCountdown] = useState(0);
  const [isDone, setIsDone] = useState(false);

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  const startCountdown = useCallback(() => {
    setCountdown(RESEND_COOLDOWN_SECONDS);
    if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    timerRef.current = setInterval(() => {
      setCountdown((previous) => {
        if (previous <= 1) {
          if (timerRef.current) {
            clearInterval(timerRef.current);
            timerRef.current = null;
          }
          return 0;
        }
        return previous - 1;
      });
    }, 1000);
  }, []);

  const handleSendCode = useCallback(async () => {
    const emailProblem = validateEmailDomain(email);
    setEmailError(emailProblem);
    if (emailProblem) {
      return;
    }

    setIsSendingCode(true);
    setNotice(null);
    setFormError(null);
    try {
      const result = await sendPasswordResetCode(email.trim().toLowerCase());
      // 仅 ENVIRONMENT=development 时后端会回显 debug_code（与 SMTP 是否接入无关），用于本地联调
      if (result.debug_code) {
        setCode(result.debug_code);
        setNotice({
          tone: "success",
          text: "开发模式：验证码已自动填入（生产环境将发送到邮箱）",
        });
      } else {
        setNotice({
          tone: "success",
          text: "验证码已发送，请查收邮箱（5 分钟内有效）",
        });
      }
      startCountdown();
    } catch (error) {
      setNotice({
        tone: "error",
        text:
          error instanceof ApiError
            ? error.message
            : "验证码发送失败，请稍后重试",
      });
    } finally {
      setIsSendingCode(false);
    }
  }, [email, startCountdown]);

  const handleSubmit = useCallback(
    async (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      const emailProblem = validateEmailDomain(email);
      const codeProblem = validateVerificationCode(code);
      const passwordProblem = validatePassword(password);
      setEmailError(emailProblem);
      setCodeError(codeProblem);
      setPasswordError(passwordProblem);
      if (emailProblem || codeProblem || passwordProblem) {
        return;
      }

      setIsSubmitting(true);
      setFormError(null);
      try {
        await resetPassword(email.trim().toLowerCase(), code.trim(), password);
        setIsDone(true);
      } catch (error) {
        setFormError(
          error instanceof ApiError ? error.message : "重置失败，请稍后重试",
        );
      } finally {
        setIsSubmitting(false);
      }
    },
    [code, email, password],
  );

  if (isDone) {
    return (
      <AuthShell
        title="密码已重置"
        description="为了账号安全，重置密码后该账号此前签发的所有登录会话都已立即失效。请使用新密码重新登录。"
        footer={<AuthFooterLinkToLogin />}
      >
        <h2 className="font-serif text-xl text-ink-900">重置成功</h2>
        <p className="prose-legal mt-3 text-sm text-ink-600">
          新密码已经生效，旧密码与旧登录状态均已失效。
        </p>
        <Link
          href="/login"
          className="mt-6 inline-flex w-full items-center justify-center rounded-md bg-seal-600 px-4 py-2.5 text-sm font-medium text-white transition-colors duration-150 hover:bg-seal-700"
        >
          前往登录
        </Link>
      </AuthShell>
    );
  }

  return (
    <AuthShell
      title="找回账号访问"
      description="通过注册邮箱接收验证码即可设置新密码。重置操作会同时使该账号所有已登录设备退出，属于安全设计，不是异常。"
      footer={<AuthFooterLinkToLogin />}
    >
      <h2 className="font-serif text-xl text-ink-900">重置密码</h2>
      <p className="mt-1 text-sm text-ink-500">
        验证邮箱后设置新密码
      </p>

      <form className="mt-6 space-y-4" onSubmit={handleSubmit} noValidate>
        {formError ? <Notice tone="error">{formError}</Notice> : null}
        {notice ? <Notice tone={notice.tone}>{notice.text}</Notice> : null}

        <Field
          label="邮箱"
          type="email"
          autoComplete="email"
          placeholder="yourname@qq.com"
          value={email}
          error={emailError}
          hint="仅支持 @qq.com 和 @foxmail.com"
          onChange={(event) => {
            setEmail(event.target.value);
            setEmailError(null);
            setFormError(null);
          }}
        />

        <Field
          label="验证码"
          inputMode="numeric"
          autoComplete="one-time-code"
          placeholder="6 位数字"
          maxLength={6}
          value={code}
          error={codeError}
          onChange={(event) => {
            setCode(event.target.value.replace(/\D/g, ""));
            setCodeError(null);
            setFormError(null);
          }}
          addon={
            <Button
              type="button"
              variant="secondary"
              onClick={handleSendCode}
              disabled={isSendingCode || countdown > 0}
              className="shrink-0 whitespace-nowrap px-3 text-xs"
            >
              {countdown > 0
                ? `${countdown}s 后重发`
                : isSendingCode
                  ? "发送中…"
                  : "发送验证码"}
            </Button>
          }
        />

        <Field
          label="新密码"
          type="password"
          autoComplete="new-password"
          placeholder="至少 8 位"
          value={password}
          error={passwordError}
          hint="8–128 位"
          onChange={(event) => {
            setPassword(event.target.value);
            setPasswordError(null);
            setFormError(null);
          }}
        />

        <Button type="submit" fullWidth disabled={isSubmitting}>
          {isSubmitting ? "提交中…" : "重置密码"}
        </Button>
      </form>
    </AuthShell>
  );
}

function AuthFooterLinkToLogin() {
  return (
    <span>
      想起密码了？
      <Link
        href="/login"
        className="ml-1 font-medium text-seal-600 underline-offset-4 hover:underline"
      >
        返回登录
      </Link>
    </span>
  );
}
