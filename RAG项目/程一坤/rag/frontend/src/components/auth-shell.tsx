"use client";

/**
 * 认证页面的公共外壳与逻辑钩子。
 *
 * 三个认证页（登录、注册、重置密码）共用同一套版式与提交状态机，
 * 差异只在字段数量和调用的接口 —— 因此把公共部分抽到这里，
 * 三个页面各自只写自己的字段与提交逻辑。
 */

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState, type ReactNode } from "react";

import { ApiError } from "@/lib/api-client";
import { readAccessToken, saveAccessToken } from "@/lib/auth-token";

/** 认证页版式：左侧品牌陈述，右侧表单卡片 */
export function AuthShell({
  title,
  description,
  children,
  footer,
}: {
  title: string;
  description: string;
  children: ReactNode;
  footer: ReactNode;
}) {
  return (
    <div className="mx-auto flex w-full max-w-shell flex-1 items-center px-6 py-12">
      <div className="grid w-full gap-12 lg:grid-cols-[1fr_minmax(0,26rem)] lg:gap-20">
        <section className="flex flex-col justify-center">
          <p className="mb-3 font-serif text-xs tracking-[0.3em] text-seal-600">
            法律信息检索
          </p>
          <h1 className="font-serif text-3xl leading-snug text-ink-900 lg:text-4xl">
            {title}
          </h1>
          <p className="prose-legal mt-5 max-w-md text-sm text-ink-600">
            {description}
          </p>

          <dl className="mt-10 space-y-4 border-l-2 border-ink-200 pl-5">
            {[
              { term: "法源可溯", detail: "每条结论标注引用编号，可展开查看法条原文" },
              { term: "时效过滤", detail: "按适用时间点筛选当时有效的法规版本" },
              { term: "范围明确", detail: "仅覆盖中国大陆全国层面劳动法，超范围会明确告知" },
            ].map((item) => (
              <div key={item.term}>
                <dt className="text-sm font-medium text-ink-800">{item.term}</dt>
                <dd className="mt-0.5 text-xs leading-relaxed text-ink-500">
                  {item.detail}
                </dd>
              </div>
            ))}
          </dl>
        </section>

        <section className="flex items-center">
          <div className="w-full rounded-lg border border-ink-200 bg-white p-7">
            {children}
            <div className="mt-6 border-t border-ink-100 pt-5 text-center text-sm text-ink-500">
              {footer}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

/** 认证页表单提交状态机 */
export interface AuthSubmitState {
  isSubmitting: boolean;
  /** 请求级错误，展示在表单顶部 */
  formError: string | null;
  submit: (action: () => Promise<string>) => Promise<void>;
  resetError: () => void;
}

/**
 * 统一处理「提交中 / 失败提示 / 成功后落令牌并跳转」。
 *
 * action 返回成功提示或空串；抛 ApiError 时取其 message 作为表单错误。
 */
export function useAuthSubmit(redirectTo = "/chat"): AuthSubmitState {
  const router = useRouter();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  const submit = useCallback(
    async (action: () => Promise<string>) => {
      setIsSubmitting(true);
      setFormError(null);
      try {
        const token = await action();
        if (token) {
          saveAccessToken(token);
          router.push(redirectTo);
        }
      } catch (error) {
        setFormError(
          error instanceof ApiError
            ? error.message
            : "操作失败，请稍后重试",
        );
      } finally {
        setIsSubmitting(false);
      }
    },
    [redirectTo, router],
  );

  return {
    isSubmitting,
    formError,
    submit,
    resetError: useCallback(() => setFormError(null), []),
  };
}

/**
 * 已登录用户访问认证页时直接送去问答页。
 *
 * 在客户端做而不是中间件：令牌存在 localStorage，服务端读不到，
 * 用 middleware 判断反而会先渲染一次未登录页再跳转，出现闪烁。
 */
export function useRedirectIfAuthenticated(): boolean {
  const router = useRouter();
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    if (readAccessToken()) {
      router.replace("/chat");
      return;
    }
    setChecked(true);
  }, [router]);

  return checked;
}

/** 认证页底部的切换链接 */
export function AuthFooterLink({
  text,
  linkText,
  href,
}: {
  text: string;
  linkText: string;
  href: string;
}) {
  return (
    <span>
      {text}
      <Link
        href={href}
        className="ml-1 font-medium text-seal-600 underline-offset-4 hover:underline"
      >
        {linkText}
      </Link>
    </span>
  );
}
