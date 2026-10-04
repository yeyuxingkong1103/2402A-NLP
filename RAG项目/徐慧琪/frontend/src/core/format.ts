// 引用格式化与复制文本（FR-7.4）。纯字符串函数、不碰 DOM：两个入口与页面都从
// 这里取同一份格式，复制进文书的字面量只有一个来源。
// Citation 用 import type：编译期擦除，不会把 public.ts 的运行时代码拖进模块图。
// 结构不变式是**单向**的（Task 6 审查裁定，Task 7 已按同口径改 public.ts/lawyer.ts）：
// 公众入口的模块图里不存在 lawyer.ts；反向不成立——律师入口读 public.ts 的
// nav/article 是允许的。
import type { Citation } from '@/core/api/public'
import { DEFAULT_LAW_ID, LAW_NAME } from '@/core/config'

// 条号三形态归一。先例是 tools/ask.py 的 _citation_label，规则逐条一致：
// 契约允许「第五百八十四条」全称与「584」裸数字两种写法（backend/app/generation/
// schema.py），「第584」这类漏尾字的半截写法也出现过。已带「第」照原样、否则补
// 「第」；再保证以「条」收尾——分两步补，半截写法才不会被包成「第第584条」
// （tools/tests/test_ask.py::test_format_citation_keeps_chinese_article_without_
// double_wrapping 就是为这两个坏标本立的案）。此前前端假定 article 恒为阿拉伯
// 数字：全称被包成「第第五百八十四条条」，URL 也拼出中文条号（后端实测 404）。
export function articleLabel(article: string | number): string {
  const raw = String(article).trim()
  const label = raw.startsWith('第') ? raw : `第${raw}`
  return label.endsWith('条') ? label : `${label}条`
}

// 中文数字→整数：语义逐条照抄 backend/app/ingest/cn_num.py 的 cn2int——零一二三四
// 五六七八九十百千、「两」=2、单独的「十」=10、上限千位（民法典条号 ≤1260）。
// 唯一偏离：未知字符返回 NaN 而不是静默忽略——前端拿它当「解析失败」信号，
// 与「不许静默生成坏 URL」同一笔账（见 citationUrl）。
const DIGITS = '零一二三四五六七八九'

function cn2int(text: string): number {
  if (text === '十') return 10
  let total = 0
  let num = 0
  for (const ch of text) {
    const d = DIGITS.indexOf(ch)
    if (d >= 0) num = d
    else if (ch === '两') num = 2
    else if (ch === '十') { total += 10 * (num || 1); num = 0 }
    else if (ch === '百') { total += 100 * (num || 1); num = 0 }
    else if (ch === '千') { total += 1000 * (num || 1); num = 0 }
    else return Number.NaN
  }
  return total + num
}

// 条号→阿拉伯数，给 URL 路径段用。后端实测只认阿拉伯数（arabic 579→200、
// chinese→404，人工验收首曝），所以全称必须在这里转掉。壳与 articleLabel 同一套：
// 去掉「第」「条」后，ASCII 数字直接 Number，中文数字走 cn2int；解析不出
// （空串、「附则」这类非条号、0/负数/非整数）返回 NaN，由 citationUrl 决定兜底
// 动作——调用方拿到的永远不会是一个假条号。
export function articleNo(article: string | number): number {
  if (typeof article === 'number') {
    return Number.isInteger(article) && article > 0 ? article : Number.NaN
  }
  const raw = article.trim().replace(/^第/, '').replace(/条$/, '')
  if (/^\d+$/.test(raw)) return Number(raw)
  const n = cn2int(raw)
  return n > 0 ? n : Number.NaN
}

// 全称/半截/裸数字三种写法统一渲染成「第…条」（先例见 articleLabel），中文条号
// 不再被包两遍。款/项为空时不占位：!= null 同时挡 null 与 undefined，契约给 null，
// 但类型上的可选性不该让「第undefined款」这类字面量漏进复制文本。
export function formatCitation(c: Citation): string {
  let ref = articleLabel(c.article)
  if (c.paragraph != null) ref += `（第${c.paragraph}款）`
  if (c.item != null) ref += `（第${c.item}项）`
  return `${LAW_NAME}${ref}：“${c.quote}”`
}

// FR-7.4 的复制文本：解读正文一段、空行、每条依据一行。空行是两段之间唯一的
// 分隔——少一个换行，「正文最后一行」与「第一条依据」会粘成一句。
export function buildCopyText(answer: string, citations: Citation[]): string {
  return [answer, citations.map(formatCitation).join('\n')].join('\n\n')
}

// 引用跳转的 URL 形状（设计 §四 的 `/law/:lawId/articles/:no`）。条号编码成
// 路径段——值是数据，一个带斜杠的值不能把路径劈成两段（与 api/public.ts 的
// article() 同一笔账；lawId 是常量，无需编码）。articleNo 解析失败时**不拼中文
// 路径**（后端必 404，且会被 request.ts 当普通错误吞掉，人只看到「加载失败」），
// 改为告警 + 落到两入口都有的导航页 /nav，让人还能自己翻到目标条文；
// 告警保证这不是静默降级。
export function citationUrl(c: Citation): string {
  const no = articleNo(c.article)
  if (!Number.isInteger(no) || no <= 0) {
    console.warn(`citationUrl: 条号 ${JSON.stringify(c.article)} 无法解析成阿拉伯数，回退到 /nav`)
    return '/nav'
  }
  return `/law/${DEFAULT_LAW_ID}/articles/${no}`
}
