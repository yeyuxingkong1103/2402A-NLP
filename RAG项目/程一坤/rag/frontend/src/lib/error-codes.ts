/**
 * 与后端 app/errors.py、docs/接口文档.md 2.4 错误码表严格对应的错误码常量。
 *
 * 前端只做「按码给出用户可读提示」这一件事，绝不改写后端返回的 message，
 * 因为后端 message 是权威文案，前端二次加工会两边不一致。
 */

export const ErrorCode = {
  /** 请求参数错误 */
  BAD_REQUEST: 40000,
  /** 资源不存在 */
  NOT_FOUND: 40001,
  /** 文件格式不支持 */
  UNSUPPORTED_FILE_FORMAT: 40002,
  /** 文件大小超限 */
  FILE_TOO_LARGE: 40003,
  /** 索引任务失败 */
  INDEX_FAILED: 40004,
  /** 未认证、会话令牌无效或已过期 */
  UNAUTHORIZED: 40100,
  /** 无权限 */
  FORBIDDEN: 40300,
  /** 路径不存在 */
  PATH_NOT_FOUND: 40400,
  /** 资源冲突或重复操作 */
  CONFLICT: 40900,
  /** 请求频率超限 */
  RATE_LIMIT: 42900,
  /** 系统内部错误 */
  INTERNAL_ERROR: 50000,
  /** 模型服务不可用 */
  MODEL_UNAVAILABLE: 50001,
  /** 向量数据库不可用 */
  VECTOR_DB_UNAVAILABLE: 50002,
  /** 缓存或记忆服务不可用 */
  CACHE_UNAVAILABLE: 50003,
} as const;

/** 错误码取值联合类型 */
export type ErrorCodeValue = (typeof ErrorCode)[keyof typeof ErrorCode];

/**
 * 会话失效类错误码：出现这些码时前端必须清空本地令牌并跳回登录页。
 * Redis 不可用时后端统一返回 40100（见 app/auth/current_user.py），
 * 因此不会出现「服务端故障却被当成未登录」以外的歧义。
 */
export const SESSION_INVALID_CODES: readonly number[] = [
  ErrorCode.UNAUTHORIZED,
];

/**
 * 把后端错误码翻译成给用户看的一句话。
 *
 * 只处理「前端能给出更好建议」的码；其余一律回落到后端 message，
 * 保证不出现「前端说参数错误、后端说验证码无效」这类矛盾。
 */
export function describeErrorCode(code: number, fallbackMessage: string): string {
  switch (code) {
    case ErrorCode.UNAUTHORIZED:
      return "登录状态已失效，请重新登录";
    case ErrorCode.RATE_LIMIT:
      return "操作过于频繁，请稍后再试";
    case ErrorCode.CONFLICT:
      return fallbackMessage || "该资源已存在或操作重复";
    case ErrorCode.MODEL_UNAVAILABLE:
      return "模型服务暂时不可用，请稍后重试";
    case ErrorCode.VECTOR_DB_UNAVAILABLE:
      return "知识库检索服务暂时不可用，请稍后重试";
    case ErrorCode.CACHE_UNAVAILABLE:
      return "会话服务暂时不可用，请稍后重试";
    case ErrorCode.INTERNAL_ERROR:
      return "服务出现异常，请稍后重试";
    default:
      return fallbackMessage || "请求失败，请稍后重试";
  }
}
