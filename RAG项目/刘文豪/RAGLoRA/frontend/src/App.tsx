/**
 * RAGLoRA · 多用户多角色 RAG 角色扮演系统
 *
 * 页面：登录 → 对话（角色选择 + 会话 + 链路看板）/ 检索调试台 / 知识库 / 系统状态
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  api, clearToken, getToken, setToken,
  type Character, type CollectionInfo, type Conversation, type KbDocument,
  type Message, type SearchHit, type SearchResult, type SourceRef, type Trace, type User,
} from './api'

type Page = 'chat' | 'search' | 'kb' | 'status'

// ================================================================ 通用组件
function useToast() {
  const [msg, setMsg] = useState<{ text: string; error: boolean } | null>(null)
  const show = useCallback((text: string, error = false) => {
    setMsg({ text, error })
    window.setTimeout(() => setMsg(null), 3200)
  }, [])
  const node = msg ? (
    <div className={`toast ${msg.error ? 'toast-error' : ''}`}>{msg.text}</div>
  ) : null
  return { show, node }
}

function Empty({ icon, title, desc }: { icon: string; title: string; desc?: string }) {
  return (
    <div className="empty-state">
      <div style={{ fontSize: 40, marginBottom: 10 }}>{icon}</div>
      <div style={{ fontWeight: 600, marginBottom: 6 }}>{title}</div>
      {desc && <div style={{ fontSize: 13, color: 'var(--ink-3)' }}>{desc}</div>}
    </div>
  )
}

function Spinner({ text }: { text: string }) {
  return (
    <div className="loading-row">
      <span className="loading-spinner" />
      {text}
    </div>
  )
}

function ScoreBar({ value, label }: { value: number; label: string }) {
  // RRF 分与精排分量纲不同，分别归一化展示
  const pct = Math.max(2, Math.min(100, label === '精排' ? (value + 12) / 14 * 100 : value * 100))
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
      <span style={{ width: 34, color: 'var(--ink-3)' }}>{label}</span>
      <div style={{ flex: 1, height: 5, background: 'var(--line-2)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: 'var(--brand)', borderRadius: 3 }} />
      </div>
      <span style={{ width: 62, textAlign: 'right', fontVariantNumeric: 'tabular-nums', color: 'var(--ink-2)' }}>
        {value.toFixed(3)}
      </span>
    </div>
  )
}

// ================================================================ 链路看板
function TracePanel({ trace, sources, loading }: {
  trace: Trace | null
  sources: SourceRef[]
  loading: boolean
}) {
  const steps = [
    { key: 'rewrite', label: '查询改写', ms: trace?.rewrite_ms, note: trace?.rewrite_skipped ? '已跳过（问句完整）' : undefined },
    { key: 'recall', label: '混合检索', ms: trace?.recall_ms, note: trace?.recall != null ? `召回 ${trace.recall} 条` : undefined },
    { key: 'rerank', label: '精排', ms: trace?.rerank_ms, note: trace?.reranked != null ? `精选 ${trace.reranked} 条` : undefined },
    { key: 'gen', label: '生成', ms: trace?.generate_ms },
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div className="card">
        <div className="card-header">
          <div>
            <div style={{ fontWeight: 600 }}>检索链路</div>
            <div style={{ fontSize: 12, color: 'var(--ink-3)' }}>
              {trace?.total_ms ? `本轮总耗时 ${(trace.total_ms / 1000).toFixed(1)}s` : '等待提问'}
            </div>
          </div>
          {loading && <span className="loading-spinner" />}
        </div>
        <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {trace?.rewritten_query && (
            <div style={{
              padding: '8px 10px', background: 'var(--bg-2)', borderRadius: 8,
              borderLeft: '3px solid var(--brand)', fontSize: 12.5,
            }}>
              <div style={{ color: 'var(--ink-3)', marginBottom: 3 }}>改写后查询</div>
              <div>{trace.rewritten_query}</div>
            </div>
          )}
          {steps.map((s) => (
            <div key={s.key} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12.5 }}>
              <span style={{
                width: 7, height: 7, borderRadius: '50%',
                background: s.ms != null ? 'var(--ok)' : 'var(--line)',
              }} />
              <span style={{ flex: 1 }}>{s.label}</span>
              <span style={{ color: 'var(--ink-3)' }}>{s.note || ''}</span>
              <span style={{
                width: 56, textAlign: 'right', fontVariantNumeric: 'tabular-nums',
                color: s.ms != null ? 'var(--ink-2)' : 'var(--ink-3)',
              }}>
                {s.ms != null ? `${s.ms}ms` : '—'}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div style={{ fontWeight: 600 }}>精排结果 · {sources.length} 条</div>
        </div>
        <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {sources.length === 0 && <div style={{ fontSize: 12.5, color: 'var(--ink-3)' }}>暂无来源</div>}
          {sources.map((s) => (
            <div key={s.idx} style={{
              padding: 10, background: 'var(--bg-2)', borderRadius: 8,
              border: '1px solid var(--line-2)',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                <span className="score-tag" style={{ background: 'var(--brand-soft)', color: 'var(--brand-weak)' }}>
                  [{s.idx}]
                </span>
                <span style={{ fontSize: 12, color: 'var(--ink-2)', flex: 1 }}>{s.source}</span>
              </div>
              {s.rerank_score != null && (
                <div style={{ marginBottom: 6 }}>
                  <ScoreBar value={s.rerank_score} label="精排" />
                </div>
              )}
              {s.score != null && <ScoreBar value={s.score} label="融合" />}
              <div style={{
                fontSize: 12, color: 'var(--ink-3)', marginTop: 7, lineHeight: 1.6,
                maxHeight: 62, overflow: 'hidden',
              }}>
                {s.text.slice(0, 120)}…
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ================================================================ 登录页
function LoginPage({ onDone }: { onDone: (u: User) => void }) {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  const submit = async () => {
    if (!username.trim() || !password) return setErr('请填写用户名和密码')
    setBusy(true)
    setErr('')
    try {
      const r = mode === 'login'
        ? await api.login(username.trim(), password)
        : await api.register(username.trim(), password)
      setToken(r.access_token)
      onDone(r.user)
    } catch (e) {
      setErr(e instanceof Error ? e.message : '请求失败')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="aurora-bg" style={{
      minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}>
      <div className="card" style={{ width: 400 }}>
        <div className="card-body" style={{ padding: '34px 34px 30px' }}>
          <div style={{ textAlign: 'center', marginBottom: 26 }}>
            <div style={{ fontSize: 34 }}>🩺⚖️</div>
            <div style={{ fontSize: 21, fontWeight: 700, marginTop: 8 }}>RAGLoRA</div>
            <div style={{ fontSize: 13, color: 'var(--ink-2)', marginTop: 4 }}>
              多用户 · 多角色 · RAG 角色扮演系统
            </div>
          </div>

          <div style={{ display: 'flex', gap: 6, marginBottom: 18, background: 'var(--bg-2)', padding: 4, borderRadius: 10 }}>
            {(['login', 'register'] as const).map((m) => (
              <button key={m} onClick={() => { setMode(m); setErr('') }}
                className="btn btn-sm"
                style={{
                  flex: 1, justifyContent: 'center',
                  background: mode === m ? 'var(--brand)' : 'transparent',
                  color: mode === m ? '#fff' : 'var(--ink-2)',
                }}>
                {m === 'login' ? '登录' : '注册'}
              </button>
            ))}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <input className="input" placeholder="用户名（至少 3 位）" value={username}
              onChange={(e) => setUsername(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && submit()} />
            <input className="input" type="password" placeholder="密码（至少 6 位）" value={password}
              onChange={(e) => setPassword(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && submit()} />
            {err && <div style={{ fontSize: 12.5, color: 'var(--bad)' }}>{err}</div>}
            <button className="btn btn-primary" disabled={busy} onClick={submit}
              style={{ justifyContent: 'center', marginTop: 4 }}>
              {busy ? '请稍候…' : mode === 'login' ? '登 录' : '注 册'}
            </button>
          </div>

          <div style={{ fontSize: 12, color: 'var(--ink-3)', marginTop: 18, textAlign: 'center', lineHeight: 1.7 }}>
            登录后可与「心血管内科医生」「执业律师」对话<br />
            每个角色拥有独立知识库与人格提示词
          </div>
        </div>
      </div>
    </div>
  )
}

// ================================================================ 对话页
function ChatPage({ toast }: { toast: (t: string, e?: boolean) => void }) {
  const [characters, setCharacters] = useState<Character[]>([])
  const [activeChar, setActiveChar] = useState<Character | null>(null)
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeConv, setActiveConv] = useState<Conversation | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [trace, setTrace] = useState<Trace | null>(null)
  const [sources, setSources] = useState<SourceRef[]>([])
  const bottomRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  // 载入角色
  useEffect(() => {
    api.characters.list()
      .then((cs) => {
        setCharacters(cs)
        if (cs.length && !activeChar) setActiveChar(cs[0])
      })
      .catch((e) => toast(e.message, true))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 角色切换 → 载入该角色的会话
  useEffect(() => {
    if (!activeChar) return
    api.conversations.list(activeChar.id)
      .then((cs) => {
        setConversations(cs)
        setActiveConv(cs[0] || null)
      })
      .catch(() => setConversations([]))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeChar?.id])

  // 会话切换 → 载入消息
  useEffect(() => {
    if (!activeConv) {
      setMessages([]); setTrace(null); setSources([])
      return
    }
    api.conversations.messages(activeConv.id)
      .then((ms) => {
        setMessages(ms)
        const last = [...ms].reverse().find((m) => m.role === 'assistant')
        setSources(last?.sources_json || [])
        setTrace(null)
      })
      .catch((e) => toast(e.message, true))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeConv?.id])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, streaming])

  const newConversation = async () => {
    if (!activeChar) return
    try {
      const c = await api.conversations.create(activeChar.id)
      setConversations((prev) => [c, ...prev])
      setActiveConv(c)
    } catch (e) {
      toast(e instanceof Error ? e.message : '新建失败', true)
    }
  }

  const removeConversation = async (id: number) => {
    try {
      await api.conversations.remove(id)
      setConversations((prev) => prev.filter((c) => c.id !== id))
      if (activeConv?.id === id) setActiveConv(null)
    } catch (e) {
      toast(e instanceof Error ? e.message : '删除失败', true)
    }
  }

  const send = async () => {
    const q = input.trim()
    if (!q || streaming || !activeChar) return
    if (!activeConv) {
      toast('请先新建会话')
      return
    }

    setInput('')
    setStreaming(true)
    setTrace(null)
    setSources([])

    const userMsg: Message = {
      id: Date.now(), role: 'user', content: q, rewritten_query: null,
      sources_json: null, latency_ms: null, created_at: new Date().toISOString(),
    }
    const placeholder: Message = {
      id: Date.now() + 1, role: 'assistant', content: '',
      rewritten_query: null, sources_json: null, latency_ms: null,
      created_at: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, userMsg, placeholder])

    const controller = new AbortController()
    abortRef.current = controller

    try {
      await api.chat.stream(activeConv.id, q, (event, data) => {
        if (event === 'trace') {
          setTrace(data as unknown as Trace)
        } else if (event === 'sources') {
          setSources((data as unknown) as SourceRef[])
        } else if (event === 'delta') {
          const t = (data as { text: string }).text
          setMessages((prev) => {
            const next = [...prev]
            const last = next[next.length - 1]
            next[next.length - 1] = { ...last, content: last.content + t }
            return next
          })
        } else if (event === 'done') {
          const d = data as { answer: string; message_id: number }
          setMessages((prev) => {
            const next = [...prev]
            const last = next[next.length - 1]
            next[next.length - 1] = { ...last, id: d.message_id || last.id, content: d.answer }
            return next
          })
          setConversations((prev) => prev.map((c) =>
            c.id === activeConv.id
              ? { ...c, message_count: c.message_count + 2, last_message_at: new Date().toISOString() }
              : c))
        } else if (event === 'error') {
          toast((data as { message: string }).message || '生成失败', true)
        }
      }, controller.signal)
    } catch (e) {
      if ((e as Error).name !== 'AbortError') {
        toast(e instanceof Error ? e.message : '请求失败', true)
      }
    } finally {
      setStreaming(false)
      abortRef.current = null
    }
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '248px minmax(0,1fr) 330px', gap: 16, height: '100%' }}>
      {/* ---------- 左栏：角色 + 会话 ---------- */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12, minHeight: 0 }}>
        <div className="card">
          <div className="card-header"><div style={{ fontWeight: 600, fontSize: 13 }}>选择角色</div></div>
          <div className="card-body" style={{ padding: 10, display: 'flex', flexDirection: 'column', gap: 6 }}>
            {characters.map((c) => (
              <button key={c.id} onClick={() => setActiveChar(c)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 10, padding: '9px 10px',
                  borderRadius: 9, cursor: 'pointer', textAlign: 'left', width: '100%',
                  border: `1px solid ${activeChar?.id === c.id ? 'var(--brand-border)' : 'transparent'}`,
                  background: activeChar?.id === c.id ? 'var(--brand-soft)' : 'transparent',
                  color: 'inherit',
                }}>
                <span style={{ fontSize: 19 }}>{c.avatar || '🙂'}</span>
                <span style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 13, fontWeight: 600 }}>{c.name}</div>
                  <div style={{ fontSize: 11, color: 'var(--ink-3)' }}>{c.kb_collection}</div>
                </span>
              </button>
            ))}
          </div>
        </div>

        <div className="card" style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
          <div className="card-header">
            <div style={{ fontWeight: 600, fontSize: 13 }}>我的会话</div>
            <button className="btn btn-ghost btn-sm" onClick={newConversation}>+ 新建</button>
          </div>
          <div className="card-body" style={{ padding: 8, overflowY: 'auto', flex: 1 }}>
            {conversations.length === 0 && (
              <div style={{ fontSize: 12.5, color: 'var(--ink-3)', padding: 10, textAlign: 'center' }}>
                还没有会话
              </div>
            )}
            {conversations.map((c) => (
              <div key={c.id} onClick={() => setActiveConv(c)}
                style={{
                  padding: '9px 10px', borderRadius: 8, cursor: 'pointer', marginBottom: 3,
                  background: activeConv?.id === c.id ? 'var(--brand-soft)' : 'transparent',
                  display: 'flex', alignItems: 'center', gap: 8,
                }}>
                <span style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    fontSize: 12.5, whiteSpace: 'nowrap', overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}>
                    {c.title || `会话 ${c.id}`}
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--ink-3)' }}>{c.message_count} 条消息</div>
                </span>
                <button className="link-btn" style={{ color: 'var(--ink-3)', fontSize: 15 }}
                  onClick={(e) => { e.stopPropagation(); removeConversation(c.id) }}>×</button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ---------- 中栏：对话 ---------- */}
      <div className="card" style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <div className="card-header">
          <div>
            <div style={{ fontWeight: 600 }}>
              {activeChar?.avatar} {activeChar?.name || '未选择角色'}
            </div>
            <div style={{ fontSize: 12, color: 'var(--ink-3)' }}>
              {activeChar?.description || '请选择左侧角色开始对话'}
            </div>
          </div>
        </div>

        <div className="card-body" style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
          {messages.length === 0 && (
            <Empty icon="💬" title="开始对话"
              desc={activeConv ? '提出你的问题，回答会标注知识来源' : '先新建一个会话'} />
          )}
          {messages.map((m) => (
            <div key={m.id} style={{
              display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start',
              marginBottom: 16,
            }}>
              <div style={{
                maxWidth: '82%', padding: '11px 15px', borderRadius: 13, lineHeight: 1.75,
                fontSize: 13.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                background: m.role === 'user' ? 'var(--brand)' : 'var(--bg-2)',
                color: m.role === 'user' ? '#fff' : 'var(--ink)',
                border: m.role === 'user' ? 'none' : '1px solid var(--line-2)',
              }}>
                {m.content || (streaming ? '思考中…' : '')}
                {m.role === 'assistant' && m.sources_json && m.sources_json.length > 0 && (
                  <div style={{
                    marginTop: 10, paddingTop: 9, borderTop: '1px solid var(--line-2)',
                    display: 'flex', flexWrap: 'wrap', gap: 6,
                  }}>
                    {m.sources_json.map((s) => (
                      <span key={s.idx} className="score-tag"
                        title={s.text}
                        style={{ background: 'var(--brand-soft)', color: 'var(--brand-weak)', fontSize: 11 }}>
                        [{s.idx}] {s.source}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        <div className="card-body" style={{ borderTop: '1px solid var(--line-2)' }}>
          <div className="input-area">
            <input className="input" value={input} placeholder="输入你的问题…（Enter 发送）"
              disabled={streaming || !activeConv}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }} />
            {streaming ? (
              <button className="btn btn-ghost" onClick={() => abortRef.current?.abort()}>停止</button>
            ) : (
              <button className="btn btn-primary" onClick={send} disabled={!activeConv}>发送</button>
            )}
          </div>
          <div className="help">
            {activeChar ? `${activeChar.name} · 知识库 ${activeChar.kb_collection}` : ''}
            {conversations.length > 0 && activeConv ? ` · 会话 #${activeConv.id}` : ''}
          </div>
        </div>
      </div>

      {/* ---------- 右栏：链路看板 ---------- */}
      <TracePanel trace={trace} sources={sources} loading={streaming} />
    </div>
  )
}

// ================================================================ 检索调试台
function SearchPage({ toast }: { toast: (t: string, e?: boolean) => void }) {
  const [query, setQuery] = useState('')
  const [collection, setCollection] = useState('')
  const [topK, setTopK] = useState(5)
  const [useRerank, setUseRerank] = useState(true)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<SearchResult | null>(null)
  const [collections, setCollections] = useState<CollectionInfo[]>([])

  useEffect(() => {
    api.search.collections().then(setCollections).catch(() => {})
  }, [])

  const run = async () => {
    if (!query.trim()) return
    setLoading(true)
    try {
      setResult(await api.search.run({
        query: query.trim(),
        collection: collection || undefined,
        top_k: topK,
        use_rerank: useRerank,
      }))
    } catch (e) {
      toast(e instanceof Error ? e.message : '检索失败', true)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="card">
        <div className="card-header"><div style={{ fontWeight: 600 }}>检索调试台</div></div>
        <div className="card-body">
          <div className="section-desc">
            纯检索，不调用大模型。用于对比混合检索与精排的效果。
          </div>
          <div className="input-area">
            <input className="input" value={query} placeholder="输入检索词，如：违约责任有哪些承担方式"
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && run()} />
            <button className="btn btn-primary" onClick={run} disabled={loading}>
              {loading ? '检索中…' : '检索'}
            </button>
          </div>

          <div style={{ display: 'flex', gap: 18, marginTop: 14, flexWrap: 'wrap', alignItems: 'center', fontSize: 13 }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              知识库
              <select className="input" style={{ width: 'auto', padding: '6px 10px' }}
                value={collection} onChange={(e) => setCollection(e.target.value)}>
                <option value="">全部</option>
                {collections.map((c) => (
                  <option key={c.name} value={c.name}>{c.name}（{c.points}）</option>
                ))}
              </select>
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              返回
              <input className="input" type="number" min={1} max={20} value={topK}
                style={{ width: 68, padding: '6px 10px' }}
                onChange={(e) => setTopK(Number(e.target.value) || 5)} />
              条
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
              <input type="checkbox" checked={useRerank} onChange={(e) => setUseRerank(e.target.checked)} />
              启用精排
            </label>
          </div>
        </div>
      </div>

      {loading && <Spinner text="混合检索中…" />}

      {result && !loading && (
        <div className="card">
          <div className="card-header">
            <div>
              <div style={{ fontWeight: 600 }}>命中 {result.hits.length} 条</div>
              <div style={{ fontSize: 12, color: 'var(--ink-3)' }}>
                检索库 {result.collections_searched.join('、')} · 召回 {result.recall_count} 条
                {result.used_rerank && ` · 精排 ${result.rerank_ms}ms`} · 总耗时 {result.elapsed_ms}ms
              </div>
            </div>
          </div>
          <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {result.hits.map((h: SearchHit, i) => (
              <div key={i} style={{
                padding: 14, background: 'var(--bg-2)', borderRadius: 10,
                border: '1px solid var(--line-2)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <span className="score-tag"
                    style={{ background: 'var(--brand-soft)', color: 'var(--brand-weak)' }}>#{i + 1}</span>
                  <span style={{ fontSize: 13, fontWeight: 600, flex: 1 }}>
                    {h.source_label || h.source}
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--ink-3)' }}>{h.collection}</span>
                </div>
                <div style={{ display: 'flex', gap: 20, marginBottom: 9 }}>
                  <div style={{ flex: 1 }}><ScoreBar value={h.score} label="融合" /></div>
                  {h.rerank_score != null && (
                    <div style={{ flex: 1 }}><ScoreBar value={h.rerank_score} label="精排" /></div>
                  )}
                </div>
                <div style={{ fontSize: 12.5, color: 'var(--ink-2)', lineHeight: 1.75 }}>
                  {h.text.slice(0, 260)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {!result && !loading && (
        <Empty icon="🔍" title="输入检索词开始"
          desc="支持 dense + sparse 双路召回，可选精排重排" />
      )}
    </div>
  )
}

// ================================================================ 知识库
function KbPage({ toast }: { toast: (t: string, e?: boolean) => void }) {
  const [collections, setCollections] = useState<CollectionInfo[]>([])
  const [docs, setDocs] = useState<KbDocument[]>([])
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const [cs, ds] = await Promise.all([api.kb.collections(), api.kb.documents()])
      setCollections(cs)
      setDocs(ds)
    } catch (e) {
      toast(e instanceof Error ? e.message : '加载失败', true)
    } finally {
      setLoading(false)
    }
  }, [toast])

  useEffect(() => { load() }, [load])

  const statusColor: Record<string, string> = {
    ready: 'var(--ok)', failed: 'var(--bad)',
    pending: 'var(--warn)', parsing: 'var(--warn)', indexing: 'var(--warn)',
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="metrics">
        {collections.map((c) => (
          <div key={c.name} className="metric">
            <div style={{ fontSize: 12, color: 'var(--ink-3)' }}>{c.name}</div>
            <div style={{ fontSize: 25, fontWeight: 700 }}>{c.points.toLocaleString()}</div>
            <div style={{ fontSize: 11.5, color: 'var(--ink-3)' }}>
              绑定角色：{c.characters.join('、') || '无'}
            </div>
          </div>
        ))}
        <div className="metric">
          <div style={{ fontSize: 12, color: 'var(--ink-3)' }}>文档总数</div>
          <div style={{ fontSize: 25, fontWeight: 700 }}>{docs.length}</div>
          <div style={{ fontSize: 11.5, color: 'var(--ink-3)' }}>
            就绪 {docs.filter((d) => d.status === 'ready').length}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div>
            <div style={{ fontWeight: 600 }}>文档列表</div>
            <div style={{ fontSize: 12, color: 'var(--ink-3)' }}>
              入库请调用 POST /api/kb/ingest（Qdrant 嵌入式模式需在服务内执行）
            </div>
          </div>
          <button className="btn btn-ghost btn-sm" onClick={load}>刷新</button>
        </div>
        <div className="card-body">
          {loading ? <Spinner text="加载中…" /> : (
            <div className="table-wrap">
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12.5 }}>
                <thead>
                  <tr style={{ color: 'var(--ink-3)', textAlign: 'left' }}>
                    <th style={{ padding: '8px 10px' }}>来源文件</th>
                    <th style={{ padding: '8px 10px' }}>知识库</th>
                    <th style={{ padding: '8px 10px' }}>策略</th>
                    <th style={{ padding: '8px 10px', textAlign: 'right' }}>分块</th>
                    <th style={{ padding: '8px 10px' }}>状态</th>
                  </tr>
                </thead>
                <tbody>
                  {docs.map((d) => (
                    <tr key={d.id} style={{ borderTop: '1px solid var(--line-2)' }}>
                      <td style={{ padding: '9px 10px' }}>
                        {d.source_path.split(/[\\/]/).pop()}
                      </td>
                      <td style={{ padding: '9px 10px', color: 'var(--ink-2)' }}>{d.collection}</td>
                      <td style={{ padding: '9px 10px', color: 'var(--ink-2)' }}>{d.chunk_strategy}</td>
                      <td style={{ padding: '9px 10px', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                        {d.chunk_count}
                      </td>
                      <td style={{ padding: '9px 10px' }}>
                        <span style={{ color: statusColor[d.status] || 'var(--ink-2)' }}>
                          ● {d.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {docs.length === 0 && <Empty icon="📚" title="知识库为空" />}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ================================================================ 系统状态
function StatusPage() {
  const [health, setHealth] = useState<Record<string, { ok: boolean; [k: string]: unknown }> | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.health(true).then((h) => setHealth(h.checks))
      .catch(() => setHealth(null))
      .finally(() => setLoading(false))
  }, [])

  const labels: Record<string, string> = {
    mysql: 'MySQL', redis: 'Redis', qdrant: 'Qdrant',
    ollama: 'Ollama', models: '本地模型', gpu: 'GPU',
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="card">
        <div className="card-header"><div style={{ fontWeight: 600 }}>依赖探活</div></div>
        <div className="card-body">
          {loading ? <Spinner text="检查中…" /> : !health ? (
            <Empty icon="⚠️" title="无法获取健康状态" />
          ) : (
            <div className="metrics">
              {Object.entries(health).map(([k, v]) => (
                <div key={k} className="metric">
                  <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                    <span style={{ color: v.ok ? 'var(--ok)' : 'var(--bad)', fontSize: 17 }}>●</span>
                    <span style={{ fontWeight: 600 }}>{labels[k] || k}</span>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--ink-3)', marginTop: 7, lineHeight: 1.65 }}>
                    {[
                      v.version, v.device, v.ms != null ? `${v.ms}ms` : null,
                      v.vram_free_gb != null ? `空闲 ${v.vram_free_gb}GB` : null,
                      Array.isArray(v.collections) ? (v.collections as string[]).join('、') : null,
                      v.error,
                    ].filter(Boolean).join(' · ') || '正常'}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-header"><div style={{ fontWeight: 600 }}>技术栈</div></div>
        <div className="card-body" style={{ fontSize: 12.5, lineHeight: 2, color: 'var(--ink-2)' }}>
          <div><b>嵌入模型</b>　bge-m3（本地，dense 1024维 + learned sparse 双路）</div>
          <div><b>精排模型</b>　bge-reranker-v2-m3（GPU 分时复用，用完释放显存）</div>
          <div><b>向量库</b>　　Qdrant 嵌入式（dense ∥ sparse → RRF 融合）</div>
          <div><b>生成模型</b>　Ollama qwen2.5:7b（流式）</div>
          <div><b>业务库</b>　　MySQL 8.0（用户/角色/会话/消息/知识库）　<b>短期记忆</b>　Redis LIST</div>
        </div>
      </div>
    </div>
  )
}

// ================================================================ 主应用
export default function App() {
  const [user, setUser] = useState<User | null>(null)
  const [booting, setBooting] = useState(true)
  const [page, setPage] = useState<Page>('chat')
  const { show, node: toastNode } = useToast()

  useEffect(() => {
    if (!getToken()) { setBooting(false); return }
    api.me().then(setUser).catch(() => clearToken()).finally(() => setBooting(false))
  }, [])

  const logout = () => { clearToken(); setUser(null) }

  const nav = useMemo(() => ([
    { id: 'chat' as Page, icon: '💬', label: '对话', desc: '与角色对话' },
    { id: 'search' as Page, icon: '🔍', label: '检索调试台', desc: '纯检索与精排对比' },
    { id: 'kb' as Page, icon: '📚', label: '知识库', desc: '集合与文档' },
    { id: 'status' as Page, icon: '⚙️', label: '系统状态', desc: '依赖探活' },
  ]), [])

  if (booting) {
    return <div className="aurora-bg" style={{ minHeight: '100vh', display: 'grid', placeItems: 'center' }}>
      <Spinner text="加载中…" />
    </div>
  }
  if (!user) return <><LoginPage onDone={setUser} />{toastNode}</>

  const current = nav.find((n) => n.id === page)!

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="brand-mark">🩺</div>
          <div className="brand-info">
            <div className="brand-name">RAGLoRA</div>
            <div className="brand-sub">多角色 RAG 系统</div>
          </div>
        </div>

        <nav className="sidebar-nav">
          <div className="nav-section-label">功能</div>
          {nav.map((n) => (
            <div key={n.id} className="nav-item"
              onClick={() => setPage(n.id)}
              style={{
                background: page === n.id ? 'var(--brand-soft)' : 'transparent',
                borderColor: page === n.id ? 'var(--brand-border)' : 'transparent',
                cursor: 'pointer',
              }}>
              <span className="nav-icon">{n.icon}</span>
              <span>
                <div style={{ fontSize: 13, fontWeight: 600 }}>{n.label}</div>
                <div style={{ fontSize: 11, color: 'var(--ink-3)' }}>{n.desc}</div>
              </span>
            </div>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-status">
            <div className="status-pill" style={{ marginBottom: 8 }}>
              <span style={{ color: 'var(--ok)' }}>●</span> {user.display_name || user.username}
            </div>
            <button className="btn btn-ghost btn-sm" style={{ width: '100%', justifyContent: 'center' }}
              onClick={logout}>退出登录</button>
          </div>
        </div>
      </aside>

      <main className="main-area">
        <div className="topbar">
          <div>
            <div className="topbar-title">{current.icon} {current.label}</div>
            <div className="topbar-meta">{current.desc}</div>
          </div>
          <div className="topbar-badge">RAG · bge-m3 · Qdrant · qwen2.5</div>
        </div>

        <div className="content-scroll">
          <div className="page" style={{ height: page === 'chat' ? '100%' : 'auto' }}>
            {page === 'chat' && <ChatPage toast={show} />}
            {page === 'search' && <SearchPage toast={show} />}
            {page === 'kb' && <KbPage toast={show} />}
            {page === 'status' && <StatusPage />}
          </div>
        </div>
      </main>

      {toastNode}
    </div>
  )
}
