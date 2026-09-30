/**
 * 会话令牌的本地存取。
 *
 * 设计取舍：
 * 1. 令牌存在 localStorage 而非 Cookie —— 后端用 Authorization: Bearer 头，
 *    不是 Cookie 会话，所以 HttpOnly Cookie 无从下手；
 * 2. 所有读写都做 window 存在性判断，保证 SSR 期间不抛错；
 * 3. 令牌变更通过自定义事件广播，让顶栏等组件能同步登录态，
 *    避免为了一个字符串引入全局状态库。
 */

const TOKEN_STORAGE_KEY = "legal_rag_access_token";

/** 令牌变更事件名；同页内组件订阅它来刷新登录态 */
export const TOKEN_CHANGE_EVENT = "legal-rag:token-change";

function canUseStorage(): boolean {
  return typeof window !== "undefined" && !!window.localStorage;
}

/** 读取当前令牌；未登录或 SSR 时返回 null */
export function readAccessToken(): string | null {
  if (!canUseStorage()) {
    return null;
  }
  return window.localStorage.getItem(TOKEN_STORAGE_KEY);
}

/** 保存令牌并广播变更 */
export function saveAccessToken(token: string): void {
  if (!canUseStorage()) {
    return;
  }
  window.localStorage.setItem(TOKEN_STORAGE_KEY, token);
  window.dispatchEvent(new Event(TOKEN_CHANGE_EVENT));
}

/** 清除令牌并广播变更；未登录时也广播，保证界面回到未登录态 */
export function clearAccessToken(): void {
  if (!canUseStorage()) {
    return;
  }
  window.localStorage.removeItem(TOKEN_STORAGE_KEY);
  window.dispatchEvent(new Event(TOKEN_CHANGE_EVENT));
}

/**
 * 订阅令牌变更。
 *
 * 同时监听 storage 事件：多标签页场景下另一个标签页登录/登出时，
 * 本页也能同步，否则会出现「一个标签页已登出、另一个还在发请求」。
 */
export function subscribeAccessToken(listener: () => void): () => void {
  if (typeof window === "undefined") {
    return () => {};
  }
  const handleStorage = (event: StorageEvent) => {
    if (event.key === null || event.key === TOKEN_STORAGE_KEY) {
      listener();
    }
  };
  window.addEventListener(TOKEN_CHANGE_EVENT, listener);
  window.addEventListener("storage", handleStorage);
  return () => {
    window.removeEventListener(TOKEN_CHANGE_EVENT, listener);
    window.removeEventListener("storage", handleStorage);
  };
}
