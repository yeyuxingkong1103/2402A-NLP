"use client";

/**
 * 注册页：邮箱验证码 + 密码。
 *
 * 后端契约（app/auth/router.py + schemas.py）：
 *   1. POST /auth/register/code  { email }            → 发验证码
 *   2. POST /auth/register       { email, code, password } → 返回 access_token
 *
 * 验证码有时效（后端 5 分钟），因此加上倒计时按钮，避免用户反复点击。
 * 验证码不再回显 —— 后端已修复该缺陷，页面只提示「已发送到邮箱」。
 */

import { useCallback, useEffect, useRef, useState } from "react";

import {
  AuthFooterLink,
  AuthShell,
  useAuthSubmit,
  useRedirectIfAuthenticated,
} from "@/components/auth-shell";
import { Button, Field, LoadingDots, Notice } from "@/components/ui/primitives";
import { ApiError } from "@/lib/api-client";
import {
  register,
  sendRegisterCode,
  validateEmailDomain,
  validatePassword,
  validateVerificationCode,
} from "@/lib/api-auth";

/** 验证码重发倒计时秒数；与后端验证码 5 分钟有效期匹配 */
const RESEND_COOLDOWN_SECONDS = 60;

export default function RegisterPage() {
  const ready = useRedirectIfAuthenticated();
  const { isSubmitting, formError, submit, resetError } = useAuthSubmit();

  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [password, setPassword] = useState("");

  const [emailError, setEmailError] = useState<string | null>(null);
  const [codeError, setCodeError] = useState<string | null>(null);
  const [passwordError, setPasswordError] = useState<string | null>(null);

  const [isSendingCode, setIsSendingCode] = useState(false);
  const [codeSent, setCodeSent] = useState(false);
  const [codeNotice, setCodeNotice] = useState<string | null>(null);
  const [countdown, setCountdown] = useState(0);

  // 倒计时用 ref 保存定时器，卸载时必须清理，否则路由切换后仍在 setState
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
    setCodeNotice(null);
    resetError();
    try {
      await sendRegisterCode(email.trim().toLowerCase());
      setCodeSent(true);
      setCodeNotice("验证码已发送，请查收邮箱（5 分钟内有效）");
      startCountdown();
    } catch (error) {
      setCodeNotice(
        error instanceof ApiError ? error.message : "验证码发送失败，请稍后重试",
      );
    } finally {
      setIsSendingCode(false);
    }
  }, [email, resetError, startCountdown]);

  const handleSubmit = useCallback(
    (event: React.FormEvent<HTMLFormElement>) => {
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

      void submit(async () => {
        const result = await register(
          email.trim().toLowerCase(),
          code.trim(),
          password,
        );
        return result.access_token;
      });
    },
    [code, email, password, submit],
  );

  if (!ready) {
    return (
      <div className="flex flex-1 items-center justify-center py-20">
        <LoadingDots label="正在检查登录状态" />
      </div>
    );
  }

  return (
    <AuthShell
      title="建立你的检索入口"
      description="注册后即可就劳动合同、经济补偿、工时与休假等劳动法问题提问。系统只依据知识库中的法条文本作答，检索不到依据时会直接告知而不会编造。"
      footer={<AuthFooterLink text="已有账号？" linkText="登录" href="/login" />}
    >
      <h2 className="font-serif text-xl text-ink-900">注册</h2>
      <p className="mt-1 text-sm text-ink-500">
        使用 QQ 邮箱接收验证码完成注册
      </p>

      <form className="mt-6 space-y-4" onSubmit={handleSubmit} noValidate>
        {formError ? <Notice tone="error">{formError}</Notice> : null}
        {codeNotice ? (
          <Notice tone={codeSent ? "success" : "error"}>{codeNotice}</Notice>
        ) : null}

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
            resetError();
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
            resetError();
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
          label="密码"
          type="password"
          autoComplete="new-password"
          placeholder="至少 8 位"
          value={password}
          error={passwordError}
          hint="8–128 位，建议包含字母与数字"
          onChange={(event) => {
            setPassword(event.target.value);
            setPasswordError(null);
            resetError();
          }}
        />

        <Button type="submit" fullWidth disabled={isSubmitting}>
          {isSubmitting ? "注册中…" : "注册并登录"}
        </Button>
      </form>
    </AuthShell>
  );
}
