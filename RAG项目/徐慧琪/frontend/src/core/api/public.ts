// 公众侧可读端点（设计 §三 定调 1/3）。结构不变式是**单向**的（Task 6 审查裁定）：
// 公众入口的模块图里不存在 lawyer.ts（律师侧独有端点所在的模块）——公众包里不出现
// 律师侧接口名，由产物闸门 scripts/check-public-bundle.mjs 钉住；反向不成立——
// 律师入口 import 本文件取 nav/article 是允许的（两入口共用共享端点）。
import { QA_TIMEOUT_MS } from '@/core/config'
import { request } from '@/core/api/request'
import type { components } from '@/core/api-types.gen'

// 生成的 openapi 把后端的 `list[dict]` 投影成 `{[key: string]: unknown}[]`——
// 复杂类型不趁手（读 citations[0].article 要处处断言），故按计划许可在本地补
// 与后端**同形**的窄类型：字段名照抄 schemas.py 的白名单常量（CITATION_FIELDS /
// SOURCE_FIELDS），前端不自创字段；后端改键时 tools/tests/test_export_openapi.py
// 的键集红线先响，这里的字面量再跟着改。
export interface Citation {
  article: string | number
  paragraph: string | number | null
  item: string | number | null
  quote: string
}

export interface Source {
  article_no: number
  path: string
  source: string
  rerank_score: number | null
}

// 推荐律师卡 / 费用区间：形状照 schemas.py 的 lawyers / fee_range 区块
// （与 /lawyers/recommend 的响应同形，前端两处用同一套渲染组件）
export interface LawyerCard {
  name: string
  org: string
  field: string
  contact_hint: string
  demo: boolean
}

export interface FeeRange {
  status: string
  low: number | null
  high: number | null
  unit: string | null
  charge_basis: string | null
  basis: string | null
  source_doc: string | null
  source_no: string | null
}

// 其余键原样取自生成物（7 个公共键 + cause/field）；只把四个宽类型的键换成窄类型
export type PublicQAAnswer = Omit<components['schemas']['PublicQAAnswer'],
  'citations' | 'sources' | 'lawyers' | 'fee_range'> & {
  citations: Citation[]
  sources: Source[]
  lawyers: LawyerCard[]
  fee_range: FeeRange | null
}

export type ArticleResponse = components['schemas']['ArticleResponse']
export type NavResponse = components['schemas']['NavResponse']
export type NavNode = components['schemas']['NavNode']

export function publicQa(question: string, signal?: AbortSignal): Promise<PublicQAAnswer> {
  // 超时用 QA_TIMEOUT_MS（120s）：实测一发 13~24s，15s 的默认值会把每次提问都
  // 在真链路上掐死，而表现与「服务坏了」一模一样（config.ts 里有同一笔账）
  return request('/public/qa', { method: 'POST', body: { question },
    timeoutMs: QA_TIMEOUT_MS, signal })
}

export function article(lawId: string, articleNo: string | number): Promise<ArticleResponse> {
  // 路径段逐个编码：law_id 是数据（今天只有 "minfadian"），一个带斜杠的值会把
  // 路径劈成两段、落到别的路由上——编码让「值」永远是值
  const law = encodeURIComponent(lawId)
  const no = encodeURIComponent(String(articleNo))
  return request(`/law/${law}/articles/${no}`)
}

export function nav(): Promise<NavResponse> {
  return request('/law/nav')
}
