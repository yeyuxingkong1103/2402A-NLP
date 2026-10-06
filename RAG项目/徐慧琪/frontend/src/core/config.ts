// 全前端共享的常量。数字与字符串的值都锚在后端契约（schemas.py / 交付说明）：
// 问句上限 500 与后端 MAX_QUESTION_CHARS 同值——前端先拦一道，后端仍是最终判据。
export const API_PREFIX = '/api/v1'
export const DEFAULT_TIMEOUT_MS = 15_000
// 问答实测 13~24s/发（交付说明 §四），超时必须显著大于它
export const QA_TIMEOUT_MS = 120_000
export const MAX_QUESTION_CHARS = 500
// 本系统当前只服务民法典（后端公众侧 Milvus 查询硬编码 law_id）
export const DEFAULT_LAW_ID = 'minfadian'
export const LAW_NAME = '《中华人民共和国民法典》'
export const FEE_DISCLAIMER = '参考区间，不构成报价或委托'
