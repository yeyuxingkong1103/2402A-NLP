/* ============================================================
   static/js/util.js —— 全局常量与通用工具函数

   在链路中的位置：
       本文件最先加载，为后面所有 js 文件提供基础能力。
       用普通 <script src> 引入，不涉及模块系统 —— 这里定义的 $、API、escapeHtml、
       fetchJson 等在整个页面范围内都是可见的（普通 script 共享全局作用域）。

   为什么用普通 script 而不是 ES module：
       拆分前这 30 多个函数共享同一个全局作用域。用普通 script 按顺序加载时
       依然共享同一环境，所以拆开后一行代码都不用改（不必补 import/export）——
       这是风险最低的拆法。代价是文件之间的依赖关系靠加载顺序约定，不如 ES module 显式。
   ============================================================ */

// $ = document.getElementById 的简写。全文件大量用 $('xxx') 取元素，省去冗长的调用
const $ = id => document.getElementById(id);
// API 前缀。留空表示"与页面同源"—— 后端把静态页面挂在根路径（backend/server.py 末尾的 app.mount），
// 所以前端与接口同源，不需要写完整域名、也不会遇到跨域问题。
// 若将来前后端分离部署，把这里改成后端地址即可，全文件的请求都会跟着走。
const API = '';

/* ================= 工具 ================= */
// HTML 转义，防脚本注入（XSS）。凡是把"来自后端/用户的数据"拼进 innerHTML 的地方都必须过它：
// 文档正文、文件名、检索片段、角色配置都是外部内容，
// 里面若含 <script> 或 onerror= 这类片段，直接插入就会被浏览器当代码执行。
function escapeHtml(v) {
  return String(v ?? '').replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));
}
// 统一的 JSON 请求封装：拼接 API 前缀、解析响应、把各类失败统一抛成 Error。
// 三个处理上的细节：
//   1. 先取 text 再手动 JSON.parse —— 后端出错时可能返回纯文本/HTML 错误页，
//      直接 response.json() 会抛出难以理解的解析异常，这里退化成一个可读的 error 字段
//   2. 解析失败不抛异常，而是塞进 data.error —— 让下面的 !response.ok 统一处理，
//      避免"解析失败"和"HTTP 失败"变成两条不同的报错路径
//   3. 用 !response.ok 而不是看 body 里的 ok 字段 —— 以 HTTP 状态码为准更可靠，
//      错误信息优先取后端给的 data.error，取不到才退化成 "请求失败：HTTP xxx"
async function fetchJson(url, options) {
  const response = await fetch(API + url, options);
  const text = await response.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch { data = { error: text || '响应不是有效 JSON' }; }
  if (!response.ok) throw new Error(data.error || `请求失败：HTTP ${response.status}`);
  return data;
}
// 字节数转可读体积。以 1MB 为界切换单位；b 为 0/undefined 时返回 '-' 而不是 '0 KB'，
// 让"没有数据"和"数据是 0"在界面上能区分开
function fmtSize(b) { if (!b) return '-'; return b > 1048576 ? (b/1048576).toFixed(2) + ' MB' : (b/1024).toFixed(1) + ' KB'; }
// 把文件名缩短成便于在引用标签里显示的形式：去掉扩展名、按 '_' 取最后一段、超 18 字截断。
// 用途是让引用标签（引用条）保持一行内可读，完整文件名放在 title 属性里悬停可见
function shortSrc(s) { const t = String(s || '').split('_').pop().replace(/\.pdf$/i, ''); return t.length > 18 ? t.slice(0, 18) + '…' : t; }
// 在容器里显示一条居中的空态/错误提示。所有提示文案都过 escapeHtml（文案里可能含后端返回的错误信息）
function showEmpty(el, msg) { el.innerHTML = `<div class="empty">${escapeHtml(msg)}</div>`; }
// 把答案的引用列表渲染成一排引用标签。
// 按 (来源, 页码) 去重：同一页被切成多个 chunk 都被召回时，对用户来说"这一页"只需出现一次。
// 用 Set 记录已见键、filter 保留下标首次出现的那些，从而既去重又保持原有顺序
function citeChips(cites) {
  const seen = new Set();
  return (cites || []).filter(c => { const k = c.source + '|' + c.page; if (seen.has(k)) return false; seen.add(k); return true; })
    .map(c => `<span class="cite-chip" title="${escapeHtml(c.source)} 第${escapeHtml(c.page)}页"><span class="src">${escapeHtml(shortSrc(c.source))}</span><span class="page-badge">第${escapeHtml(c.page)}页</span></span>`).join('');
}

