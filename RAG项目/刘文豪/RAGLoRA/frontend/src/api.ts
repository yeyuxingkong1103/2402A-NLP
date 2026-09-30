/**
 * RAGLoRA 后端 API 客户端
 * 鉴权：Bearer Token（存 localStorage）
 * 流式：fetch + ReadableStream 手工解析 SSE（EventSource 不支持 POST）
 */

const BASE = '/api'
const TOKEN_KEY = 'raglora_token'

let _token = localStorage.getItem(TOKEN_KEY) || ''

export function getToken() {
  return _token
}
export function setToken(t: string) {
  _token = t
  localStorage.setItem(TOKEN_KEY, t)
}
export function clearToken() {
  _token = ''
  localStorage.removeItem(TOKEN_KEY)
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...((init.headers as Record<string, string>) || {}),
  }
  if (_token) headers['Authorization'] = `Bearer ${_token}`

  const res = await fetch(BASE + path, { ...init, headers })

  if (res.status === 401) {
    clearToken()
    throw new Error('登录已过期，请重新登录')
  }
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      detail = body.detail || body.message || detail
    } catch {
      /* 响应体非 JSON，沿用状态码 */
    }
    throw new Error(detail)
  }
  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

const post = <T,>(p: string, body: unknown) =>
  request<T>(p, { method: 'POST', body: JSON.stringify(body) })
const patch = <T,>(p: string, body: unknown) =>
  request<T>(p, { method: 'PATCH', body: JSON.stringify(body) })
const put = <T,>(p: string, body: unknown) =>
  request<T>(p, { method: 'PUT', body: JSON.stringify(body) })
const del = <T,>(p: string) => request<T>(p, { method: 'DELETE' })

// ---------------------------------------------------------------- 类型
export interface User {
  id: number
  username: string
  display_name: string | null
  created_at: string
}

export interface Character {
  id: number
  slug: string
  name: string
  category: string | null
  avatar: string | null
  description: string | null
  kb_collection: string
  is_builtin: boolean
  identity_block: string | null
  style_json: Record<string, string> | null
  domain_constraints: string | null
  prompt_template: string | null
  recall_top_k: number
  rerank_top_k: number
  temperature: number
}

export interface Conversation {
  id: number
  user_id: number
  character_id: number
  title: string | null
  message_count: number
  last_message_at: string | null
  created_at: string
}

export interface SourceRef {
  idx: number
  source: string
  collection: string | null
  law_name: string | null
  article_no: string | null
  page: number | null
  score: number | null
  rerank_score: number | null
  text: string
}

export interface Message {
  id: number
  role: 'user' | 'assistant' | 'system'
  content: string
  rewritten_query: string | null
  sources_json: SourceRef[] | null
  latency_ms: number | null
  created_at: string
}

export interface Trace {
  rewritten_query?: string
  rewrite_ms?: number
  rewrite_skipped?: boolean
  recall?: number
  recall_ms?: number
  reranked?: number
  rerank_ms?: number
  total_retrieval_ms?: number
  generate_ms?: number
  total_ms?: number
}

export interface SearchHit {
  id: number | string
  score: number
  rerank_score?: number | null
  text: string
  source: string | null
  page: number | null
  law_name: string | null
  article_no: string | null
  collection: string
  source_label?: string
}

export interface SearchResult {
  query: string
  collection: string
  collections_searched: string[]
  recall_count: number
  used_rerank: boolean
  rerank_ms: number
  elapsed_ms: number
  hits: SearchHit[]
}

export interface CollectionInfo {
  name: string
  points: number
  characters: string[]
}

export interface KbDocument {
  id: number
  collection: string
  source_path: string
  doc_type: string
  chunk_strategy: string | null
  chunk_count: number
  status: string
  error_msg: string | null
  created_at: string
}

export interface HealthCheck {
  ok: boolean
  checks: Record<string, { ok: boolean; [k: string]: unknown }>
}

// ---------------------------------------------------------------- API
export const api = {
  // 鉴权
  register: (username: string, password: string, display_name?: string) =>
    post<{ access_token: string; user: User }>('/auth/register', {
      username, password, display_name,
    }),
  login: (username: string, password: string) =>
    post<{ access_token: string; user: User }>('/auth/login', { username, password }),
  me: () => request<User>('/auth/me'),

  // 角色
  characters: {
    list: () => request<Character[]>('/characters'),
    get: (id: number) => request<Character>(`/characters/${id}`),
    create: (body: Partial<Character>) => post<Character>('/characters', body),
    update: (id: number, body: Partial<Character>) => put<Character>(`/characters/${id}`, body),
    remove: (id: number) => del<{ deleted: number }>(`/characters/${id}`),
  },

  // 会话
  conversations: {
    list: (characterId?: number) =>
      request<Conversation[]>(
        `/conversations${characterId ? `?character_id=${characterId}` : ''}`,
      ),
    create: (character_id: number, title?: string) =>
      post<Conversation>('/conversations', { character_id, title }),
    messages: (id: number, limit = 100) =>
      request<Message[]>(`/conversations/${id}/messages?limit=${limit}`),
    rename: (id: number, title: string) =>
      patch<Conversation>(`/conversations/${id}`, { title }),
    remove: (id: number) => del<{ deleted: number }>(`/conversations/${id}`),
    memory: (id: number) =>
      request<{ redis_available: boolean; redis: unknown[]; from_mysql: unknown[] }>(
        `/conversations/${id}/memory`,
      ),
  },

  // 对话
  chat: {
    completions: (conversation_id: number, question: string, top_k?: number) =>
      post<{ answer: string; sources: SourceRef[]; trace: Trace; message_id: number }>(
        '/chat/completions', { conversation_id, question, top_k },
      ),
    /** 流式对话。onEvent 按事件名回调。返回可用于中断的 AbortController。 */
    stream(
      conversation_id: number,
      question: string,
      onEvent: (event: string, data: Record<string, unknown>) => void,
      signal?: AbortSignal,
    ): Promise<void> {
      return (async () => {
        const res = await fetch(`${BASE}/chat/stream`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(_token ? { Authorization: `Bearer ${_token}` } : {}),
          },
          body: JSON.stringify({ conversation_id, question }),
          signal,
        })
        if (!res.ok || !res.body) {
          throw new Error(`流式请求失败 HTTP ${res.status}`)
        }
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buf = ''
        let eventName = 'message'

        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          buf += decoder.decode(value, { stream: true })

          const blocks = buf.split('\n\n')
          buf = blocks.pop() || ''
          for (const block of blocks) {
            for (const line of block.split('\n')) {
              if (line.startsWith('event: ')) {
                eventName = line.slice(7).trim()
              } else if (line.startsWith('data: ')) {
                try {
                  onEvent(eventName, JSON.parse(line.slice(6)))
                } catch {
                  /* 忽略解析失败的分片 */
                }
              }
            }
          }
        }
      })()
    },
  },

  // 检索调试台
  search: {
    run: (body: {
      query: string
      collection?: string
      top_k?: number
      recall_k?: number
      use_rerank?: boolean
      filters?: Record<string, string>
    }) => post<SearchResult>('/search', body),
    collections: () => request<CollectionInfo[]>('/search/collections'),
    history: () => request<Array<{
      id: number; query: string; collection: string | null
      elapsed_ms: number; created_at: string; result: SearchResult
    }>>('/search/history'),
    deleteHistory: (id: number) => del<{ deleted: number }>(`/search/history/${id}`),
  },

  // 知识库
  kb: {
    ingest: (path: string, collection: string, strategy = 'auto') =>
      post<{ job_id: string; status: string }>('/kb/ingest', { path, collection, strategy }),
    job: (id: string) => request<Record<string, unknown>>(`/kb/jobs/${id}`),
    documents: (collection?: string) =>
      request<KbDocument[]>(
        `/kb/documents${collection ? `?collection=${collection}` : ''}`,
      ),
    removeDocument: (id: number) => del<{ ok: boolean }>(`/kb/documents/${id}`),
    collections: () => request<CollectionInfo[]>('/kb/collections'),
  },

  health: (deep = false) => request<HealthCheck>(`/health${deep ? '?deep=1' : ''}`),
}
