"use client";

/**
 * 登录页：邮箱 + 密码。
 *
 * 后端仅允许 qq.com / foxmail.com 邮箱（app/auth/schemas.py ALLOWED_DOMAINS），
 * 前端做前置提示，避免用户白填一遍再被拒。
 */

import Link from "next/link";
import { useCallback, useState } from "react";

import {
  AuthFooterLink,
  AuthShell,
  useAuthSubmit,
  useRedirectIfAuthenticated,
} from "@/components/auth-shell";
import { Button, Field, LoadingDots, Notice } from "@/components/ui/primitives";
import { login, validateEmailDomain } from "@/lib/api-auth";

export default function LoginPage() {
  const ready = useRedirectIfAuthenticated();
  const { isSubmitting, formError, submit, resetError } = useAuthSubmit();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [emailError, setEmailError] = useState<string | null>(null);
  const [passwordError, setPasswordError] = useState<string | null>(null);

  const handleSubmit = useCallback(
    (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      const emailProblem = validateEmailDomain(email);
      const passwordProblem = password ? null : "请输入密码";
      setEmailError(emailProblem);
      setPasswordError(passwordProblem);
      if (emailProblem || passwordProblem) {
        return;
      }

      void submit(async () => {
        const result = await login(email.trim().toLowerCase(), password);
        return result.access_token;
      });
    },
    [email, password, submit],
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
      title="以法为据，信息可溯"
      description="面向中国大陆劳动法的法律信息检索助手。回答基于官方公开法规文本，逐条标注引用，并区分法规原文、说明与待核实事实。"
      footer={
        <AuthFooterLink text="还没有账号？" linkText="注册" href="/register" />
      }
    >
      <h2 className="font-serif text-xl text-ink-900">登录</h2>
      <p className="mt-1 text-sm text-ink-500">
        使用注册时填写的 QQ 邮箱登录
      </p>

      <form className="mt-5 space-y-4" onSubmit={handleSubmit} noValidate>
        {formError ? <Notice tone="error">{formError}</Notice> : null}

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
          label="密码"
          type="password"
          autoComplete="current-password"
          placeholder="请输入密码"
          value={password}
          error={passwordError}
          onChange={(event) => {
            setPassword(event.target.value);
            setPasswordError(null);
            resetError();
          }}
        />

        <div className="text-right">
          <Link
            href="/password-reset"
            className="text-xs text-ink-500 underline-offset-4 hover:text-ink-800 hover:underline"
          >
            忘记密码？
          </Link>
        </div>

        <Button type="submit" fullWidth disabled={isSubmitting}>
          {isSubmitting ? "登录中…" : "登录"}
        </Button>
      </form>
    </AuthShell>
  );
}
