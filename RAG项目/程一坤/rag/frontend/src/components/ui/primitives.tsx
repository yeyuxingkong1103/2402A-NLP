/**
 * 通用 UI 基础组件。
 *
 * 全部保持无依赖、只有样式与语义，不引入组件库：
 * 需求文档只锁定了 Next.js + TypeScript，多加一层组件库会带来额外学习成本，
 * 而首期需要的控件数量很少，手写反而更可控。
 */

import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode } from "react";

/** 把多个 className 拼起来，自动过滤空值 */
export function cx(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(" ");
}

type ButtonVariant = "primary" | "secondary" | "ghost";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  fullWidth?: boolean;
}

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  // 印章红作主操作色，与「法律」语境呼应
  primary:
    "bg-seal-600 text-white hover:bg-seal-700 active:bg-seal-800 disabled:bg-ink-300",
  secondary:
    "bg-white text-ink-800 border border-ink-200 hover:border-ink-400 hover:bg-ink-50 disabled:text-ink-400",
  ghost: "bg-transparent text-ink-600 hover:bg-ink-100 hover:text-ink-900",
};

/** 主按钮：统一禁用态与加载态样式 */
export function Button({
  variant = "primary",
  fullWidth = false,
  className,
  children,
  ...rest
}: ButtonProps) {
  return (
    <button
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-md px-4 py-2.5",
        "text-sm font-medium transition-colors duration-150",
        "disabled:cursor-not-allowed",
        BUTTON_VARIANTS[variant],
        fullWidth && "w-full",
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

interface FieldProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
  /** 字段级错误文案；有值时输入框转为错误态 */
  error?: string | null;
  /** 字段下方的辅助说明，例如格式要求 */
  hint?: string;
  /** 右侧附加内容，例如「发送验证码」按钮 */
  addon?: ReactNode;
}

/** 表单字段：标签 + 输入框 + 错误/提示，统一三段的排布 */
export function Field({
  label,
  error,
  hint,
  addon,
  className,
  id,
  ...rest
}: FieldProps) {
  const inputId = id ?? `field-${label}`;
  const describedBy = error
    ? `${inputId}-error`
    : hint
      ? `${inputId}-hint`
      : undefined;

  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={inputId} className="text-sm font-medium text-ink-700">
        {label}
      </label>
      <div className="flex items-stretch gap-2">
        <input
          id={inputId}
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy}
          className={cx(
            "w-full rounded-md border bg-white px-3 py-2.5 text-sm text-ink-900",
            "placeholder:text-ink-400 transition-colors duration-150",
            "focus:outline-none focus-visible:outline-none",
            error
              ? "border-seal-400 focus:border-seal-500"
              : "border-ink-200 focus:border-ink-500",
            className,
          )}
          {...rest}
        />
        {addon}
      </div>
      {error ? (
        <p id={`${inputId}-error`} role="alert" className="text-xs text-seal-600">
          {error}
        </p>
      ) : hint ? (
        <p id={`${inputId}-hint`} className="text-xs text-ink-500">
          {hint}
        </p>
      ) : null}
    </div>
  );
}

/** 提示条：用于展示请求级错误或说明性信息 */
export function Notice({
  tone = "info",
  children,
}: {
  tone?: "info" | "error" | "success";
  children: ReactNode;
}) {
  const toneStyles = {
    info: "border-ink-200 bg-ink-50 text-ink-700",
    error: "border-seal-200 bg-seal-50 text-seal-700",
    success: "border-jade-200 bg-jade-50 text-jade-700",
  } as const;

  return (
    <div
      role={tone === "error" ? "alert" : undefined}
      className={cx(
        "rounded-md border px-3.5 py-2.5 text-sm leading-relaxed",
        toneStyles[tone],
      )}
    >
      {children}
    </div>
  );
}

/** 小标签：文书类型、现行有效状态等 */
export function Tag({
  tone = "neutral",
  children,
}: {
  tone?: "neutral" | "current" | "expired" | "unknown" | "accent";
  children: ReactNode;
}) {
  const toneStyles = {
    neutral: "border-ink-200 bg-ink-50 text-ink-600",
    current: "border-jade-200 bg-jade-50 text-jade-700",
    expired: "border-ink-300 bg-ink-100 text-ink-600",
    unknown: "border-ink-200 bg-white text-ink-500",
    accent: "border-seal-200 bg-seal-50 text-seal-700",
  } as const;

  return (
    <span
      className={cx(
        "inline-flex items-center rounded border px-2 py-0.5 text-xs font-medium",
        toneStyles[tone],
      )}
    >
      {children}
    </span>
  );
}

/** 加载指示：三点跳动，用于等待首字节 */
export function LoadingDots({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm text-ink-500">
      <span className="flex gap-1" aria-hidden="true">
        {[0, 1, 2].map((index) => (
          <span
            key={index}
            className="h-1.5 w-1.5 animate-pulse rounded-full bg-ink-400"
            style={{ animationDelay: `${index * 150}ms` }}
          />
        ))}
      </span>
      {label ? <span>{label}</span> : null}
    </span>
  );
}
