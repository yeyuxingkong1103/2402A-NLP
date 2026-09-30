import { useEffect, useMemo, useState } from 'react'
import './styles.css'

const suggestions = [
  '第四学段（7～9年级）的阅读教学要求是什么？',
  '请根据九年级上册教学案例设计一节阅读课。',
  '九年级语文期中试题主要考查哪些能力？',
]

const knowledgeStats = [
  { label: '知识片段', value: '1114' },
  { label: '覆盖资料', value: '20' },
  { label: '记忆模式', value: 'Redis' },
]

const capabilityCards = [
  { title: '课标解读', text: '快速定位第四学段要求，提炼教学目标与能力点。' },
  { title: '课堂设计', text: '结合教学案例生成导入、活动、板书和评价建议。' },
  { title: '试题分析', text: '从期中试题中归纳考点、题型与复习方向。' },
]

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || ''

function App() {
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [activeSession, setActiveSession] = useState('')
  const [documents, setDocuments] = useState([])
  const [documentLoading, setDocumentLoading] = useState(false)
  const [filters, setFilters] = useState({ subject: '语文', grade: '九年级', chapter: '', document_type: '' })
  const [generator, setGenerator] = useState('lesson')
  const [generationLoading, setGenerationLoading] = useState(false)
  const [generationResult, setGenerationResult] = useState(null)
  const [savedLessonPlans, setSavedLessonPlans] = useState([])
  const [savedQuestionSets, setSavedQuestionSets] = useState([])
  const [generationForm, setGenerationForm] = useState({
    chapter: '', lesson_hours: 2, requirements: '', knowledge_point: '', question_type: '选择题', question_count: 5, difficulty: '中等',
  })

  const hasMessages = messages.length > 0
  const currentTitle = useMemo(() => {
    const current = sessions.find((session) => session.id === activeSession)
    return current?.title || '新对话'
  }, [activeSession, sessions])

  useEffect(() => {
    loadSessions()
    loadDocuments()
    loadSavedContent()
  }, [])

  useEffect(() => {
    if (activeSession) loadMessages(activeSession)
  }, [activeSession])

  useEffect(() => {
    const hasPending = documents.some((document) => ['待解析', '解析中'].includes(document.status))
    if (!hasPending) return undefined
    const timer = window.setInterval(loadDocuments, 3000)
    return () => window.clearInterval(timer)
  }, [documents])

  async function request(path, options = {}) {
    let response
    try {
      response = await fetch(`${apiBaseUrl}${path}`, {
        ...(options.body instanceof FormData ? {} : { headers: { 'Content-Type': 'application/json', ...(options.headers || {}) } }),
        ...options,
      })
    } catch {
      throw new Error('后端服务未启动，请先启动 FastAPI。')
    }

    const text = await response.text()
    const data = text ? parseJson(text) : {}
    if (!response.ok) {
      throw new Error(data.detail || data.message || `服务暂时不可用（${response.status}）`)
    }
    return data
  }

  function parseJson(text) {
    try {
      return JSON.parse(text)
    } catch {
      return { detail: text || '服务返回了空响应' }
    }
  }

  async function loadSessions() {
    try {
      const data = await request('/api/sessions')
      setSessions(data.sessions || [])
      if (!activeSession && data.sessions?.length) setActiveSession(data.sessions[0].id)
    } catch (sessionError) {
      setError(sessionError.message)
    }
  }

  async function loadMessages(sessionId) {
    try {
      const data = await request(`/api/sessions/${sessionId}/messages`)
      setMessages(data.messages || [])
    } catch (messageError) {
      setError(messageError.message)
    }
  }

  async function loadDocuments() {
    try {
      const data = await request('/api/documents')
      setDocuments(data.documents || [])
    } catch (documentError) {
      setError(documentError.message)
    }
  }

  async function loadSavedContent() {
    try {
      const [lessonData, questionData] = await Promise.all([
        request('/api/lesson-plans'),
        request('/api/questions'),
      ])
      setSavedLessonPlans(lessonData.lesson_plans || [])
      setSavedQuestionSets(questionData.questions || [])
    } catch (contentError) {
      setError(contentError.message)
    }
  }

  async function saveGeneratedContent() {
    if (!generationResult) return
    const isLesson = generator === 'lesson'
    const payload = isLesson
      ? {
          subject: filters.subject || '语文', grade: filters.grade || '九年级', chapter: generationForm.chapter,
          lesson_hours: Number(generationForm.lesson_hours), requirements: generationForm.requirements,
          title: generationForm.chapter, content: generationResult.content, citations: generationResult.citations || [],
        }
      : {
          subject: filters.subject || '语文', grade: filters.grade || '九年级', chapter: generationForm.chapter,
          knowledge_point: generationForm.knowledge_point, question_type: generationForm.question_type,
          question_count: Number(generationForm.question_count), difficulty: generationForm.difficulty,
          content: generationResult.content, citations: generationResult.citations || [],
        }
    try {
      await request(isLesson ? '/api/lesson-plans' : '/api/questions', { method: 'POST', body: JSON.stringify(payload) })
      await loadSavedContent()
      setError('保存成功')
    } catch (saveError) {
      setError(saveError.message)
    }
  }

  async function uploadDocument(event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    setDocumentLoading(true)
    setError('')
    try {
      const formData = new FormData()
      formData.append('file', file)
      const data = await request('/api/documents/upload', {
        method: 'POST',
        headers: {},
        body: formData,
      })
      setDocuments((current) => [data, ...current])
    } catch (uploadError) {
      setError(uploadError.message)
    } finally {
      setDocumentLoading(false)
    }
  }

  async function deleteDocument(documentId) {
    if (!window.confirm('确定删除这份资料及其向量吗？')) return
    try {
      await request(`/api/documents/${documentId}`, { method: 'DELETE' })
      setDocuments((current) => current.filter((document) => document.id !== documentId))
    } catch (deleteError) {
      setError(deleteError.message)
    }
  }

  async function generateEducationContent(event) {
    event.preventDefault()
    setGenerationLoading(true)
    setGenerationResult(null)
    setError('')
    const endpoint = generator === 'lesson' ? '/api/lesson-plans/generate' : '/api/questions/generate'
    const body = generator === 'lesson'
      ? { subject: filters.subject || '语文', grade: filters.grade || '九年级', chapter: generationForm.chapter, lesson_hours: Number(generationForm.lesson_hours), requirements: generationForm.requirements }
      : { subject: filters.subject || '语文', grade: filters.grade || '九年级', chapter: generationForm.chapter, knowledge_point: generationForm.knowledge_point, question_type: generationForm.question_type, question_count: Number(generationForm.question_count), difficulty: generationForm.difficulty }
    try {
      const data = await request(endpoint, { method: 'POST', body: JSON.stringify(body) })
      setGenerationResult(data)
    } catch (generationError) {
      setError(generationError.message)
    } finally {
      setGenerationLoading(false)
    }
  }

  async function ensureSession(title = '新对话') {
    if (activeSession) return activeSession
    const session = await request('/api/sessions', {
      method: 'POST',
      body: JSON.stringify({ title }),
    })
    setActiveSession(session.id)
    setSessions((current) => [session, ...current])
    return session.id
  }

  async function askQuestion(nextQuestion = question) {
    const trimmedQuestion = nextQuestion.trim()
    if (!trimmedQuestion || loading) return

    setQuestion('')
    setError('')
    setMessages((current) => [...current, { role: 'user', content: trimmedQuestion }])
    setLoading(true)

    try {
      const sessionId = await ensureSession(trimmedQuestion.slice(0, 24))
      const data = await request('/api/chat', {
        method: 'POST',
        body: JSON.stringify({
          question: trimmedQuestion,
          session_id: sessionId,
          top_k: 5,
          ...Object.fromEntries(Object.entries(filters).filter(([, value]) => value)),
        }),
      })
      setMessages((current) => [
        ...current,
        { role: 'assistant', content: data.answer, citations: data.citations || [] },
      ])
      await loadSessions()
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setLoading(false)
    }
  }

  async function clearCurrentChat() {
    if (!activeSession || !window.confirm('确定清空当前对话吗？')) return
    try {
      await request(`/api/sessions/${activeSession}/messages`, { method: 'DELETE' })
      setMessages([])
      await loadSessions()
    } catch (requestError) {
      setError(requestError.message)
    }
  }

  async function deleteCurrentSession(sessionId) {
    if (!window.confirm('确定删除这条对话及其记忆吗？')) return
    try {
      await request(`/api/sessions/${sessionId}`, { method: 'DELETE' })
      const remaining = sessions.filter((session) => session.id !== sessionId)
      setSessions(remaining)
      if (sessionId === activeSession) {
        setActiveSession(remaining[0]?.id || '')
        setMessages([])
      }
    } catch (requestError) {
      setError(requestError.message)
    }
  }

  async function startNewChat() {
    try {
      setError('')
      const session = await request('/api/sessions', {
        method: 'POST',
        body: JSON.stringify({ title: '新对话' }),
      })
      setActiveSession(session.id)
      setMessages([])
      setQuestion('')
      setSessions((current) => [session, ...current])
    } catch (sessionError) {
      setError(sessionError.message)
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">文</div>
          <div>
                <div className="brand-name">文脉 <em>01</em></div>
            <div className="brand-caption">教师智能助手</div>
          </div>
        </div>

        <button className="new-chat" onClick={startNewChat}>
          <span>＋</span> 新建对话
        </button>

        <div className="side-label">Redis 记忆</div>
        <div className="session-list">
          {sessions.length === 0 && <div className="empty-session">暂无历史对话</div>}
          {sessions.map((session) => (
            <div
              className={`session-item ${session.id === activeSession ? 'active' : ''}`}
              key={session.id}
              role="button"
              tabIndex="0"
              onClick={() => setActiveSession(session.id)}
              onKeyDown={(event) => { if (event.key === 'Enter') setActiveSession(session.id) }}
            >
              <span className="session-icon">◌</span>
              <span className="session-copy">
                <strong>{session.title || '新对话'}</strong>
                <small>短期记忆已保存</small>
              </span>
              <button className="delete-session" aria-label="删除对话" onClick={(event) => { event.stopPropagation(); deleteCurrentSession(session.id) }}>×</button>
            </div>
          ))}
        </div>

        <div className="sidebar-bottom">
          <div className="knowledge-card">
            <div className="knowledge-icon">✦</div>
            <div>
              <strong>教育知识库</strong>
              <span>{documents.length} 份资料 · {documents.reduce((total, document) => total + (document.chunk_count || 0), 0)} 个片段</span>
            </div>
            <label className="upload-button">
              {documentLoading ? '上传中…' : '上传'}
              <input type="file" accept=".pdf,.doc,.docx,.md,.txt" onChange={uploadDocument} disabled={documentLoading} />
            </label>
          </div>
          <div className="document-list">
            {documents.length === 0 && <span className="document-empty">暂无上传资料</span>}
            {documents.slice(0, 5).map((document) => (
              <div className="document-item" key={document.id}>
                <span className="document-type">{document.file_type?.toUpperCase()}</span>
                <span className="document-name" title={document.file_name}>{document.file_name}</span>
                <span className={`document-status status-${document.status}`}>{document.status}</span>
                <button className="delete-document" onClick={() => deleteDocument(document.id)} aria-label="删除文档">×</button>
              </div>
            ))}
          </div>
          <div className="profile">
            <div className="avatar">李</div>
            <div><strong>李老师</strong><span>九年级语文</span></div>
            <span className="more">···</span>
          </div>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div>
            <div className="breadcrumb">编辑部工作台 <span>/</span> 文脉问答</div>
            <h1>{currentTitle}</h1>
          </div>
          <div className="top-actions">
            <select value={filters.subject} onChange={(event) => setFilters({ ...filters, subject: event.target.value })} aria-label="学科筛选">
              <option value="">全部学科</option>
              <option value="语文">语文</option>
              <option value="数学">数学</option>
              <option value="英语">英语</option>
            </select>
            <select value={filters.grade} onChange={(event) => setFilters({ ...filters, grade: event.target.value })} aria-label="年级筛选">
              <option value="">全部年级</option>
              <option value="七年级">七年级</option>
              <option value="八年级">八年级</option>
              <option value="九年级">九年级</option>
            </select>
            <select value={filters.document_type} onChange={(event) => setFilters({ ...filters, document_type: event.target.value })} aria-label="资料类型筛选">
              <option value="">全部资料</option>
              <option value="课程标准">课程标准</option>
              <option value="教案">教案</option>
              <option value="教学案例">教学案例</option>
              <option value="题库">题库</option>
            </select>
            <button className="tool-toggle" onClick={() => { setGenerator('lesson'); setGenerationResult(null) }}>教案生成</button>
            <button className="tool-toggle" onClick={() => { setGenerator('questions'); setGenerationResult(null) }}>试题生成</button>
            <span className="service-status"><i /> Redis 记忆</span>
            <button className="clear-chat" onClick={clearCurrentChat} disabled={!activeSession}>清空对话</button>
            <button className="icon-button" aria-label="帮助">?</button>
            <button className="icon-button" aria-label="设置">⚙</button>
          </div>
        </header>

        <section className="generation-panel">
          <div className="generation-heading">
            <div><strong>{generator === 'lesson' ? '教案生成' : '试题生成'}</strong><span>基于当前学科、年级筛选和知识库资料</span></div>
            {generationLoading && <small>正在生成…</small>}
          </div>
          <form className="generation-form" onSubmit={generateEducationContent}>
            <input required value={generationForm.chapter} onChange={(event) => setGenerationForm({ ...generationForm, chapter: event.target.value })} placeholder="章节，例如：九年级上册第一单元" />
            {generator === 'lesson' ? (
              <>
                <input type="number" min="1" max="8" value={generationForm.lesson_hours} onChange={(event) => setGenerationForm({ ...generationForm, lesson_hours: event.target.value })} aria-label="课时" />
                <input value={generationForm.requirements} onChange={(event) => setGenerationForm({ ...generationForm, requirements: event.target.value })} placeholder="教学要求（可选）" />
              </>
            ) : (
              <>
                <input required value={generationForm.knowledge_point} onChange={(event) => setGenerationForm({ ...generationForm, knowledge_point: event.target.value })} placeholder="知识点" />
                <select value={generationForm.question_type} onChange={(event) => setGenerationForm({ ...generationForm, question_type: event.target.value })}><option>选择题</option><option>填空题</option><option>简答题</option><option>阅读理解题</option></select>
                <input type="number" min="1" max="20" value={generationForm.question_count} onChange={(event) => setGenerationForm({ ...generationForm, question_count: event.target.value })} aria-label="题目数量" />
                <select value={generationForm.difficulty} onChange={(event) => setGenerationForm({ ...generationForm, difficulty: event.target.value })}><option>简单</option><option>中等</option><option>困难</option></select>
              </>
            )}
            <button type="submit" disabled={generationLoading}>{generationLoading ? '生成中…' : '开始生成'}</button>
          </form>
          {generationResult && <div className="generation-result"><div className="generation-content">{generationResult.content}</div><div className="generation-result-actions"><button className="save-generated" onClick={saveGeneratedContent}>保存到我的{generator === 'lesson' ? '教案' : '试题'}</button>{generationResult.citations?.length > 0 && <div className="generation-citations">引用 {generationResult.citations.length} 条知识库资料</div>}</div></div>}
          {(savedLessonPlans.length > 0 || savedQuestionSets.length > 0) && <div className="saved-content-list"><strong>我的教学资产</strong>{savedLessonPlans.slice(0, 3).map((item) => <button key={`lesson-${item.id}`} onClick={() => setGenerationResult(item)}><span>教案</span>{item.title}</button>)}{savedQuestionSets.slice(0, 3).map((item) => <button key={`questions-${item.id}`} onClick={() => setGenerationResult(item)}><span>试题</span>{item.chapter} · {item.knowledge_point}</button>)}</div>}
        </section>

        <section className={`chat-area ${hasMessages ? 'has-messages' : ''}`}>
          {!hasMessages && (
            <div className="welcome">
              <div className="welcome-orbit"><span>文</span></div>
              <p className="eyebrow">TEACHER'S DESK · 九年级语文</p>
              <h2>把每一堂课，<br /><span>讲得更有依据。</span></h2>
              <p className="welcome-text">文脉从课程标准、教学案例与试题资料中整理线索，也会记住你的近期对话，让每一次备课都有连续的思路。</p>
              <div className="stats-row">
                {knowledgeStats.map((item) => (
                  <div className="stat-card" key={item.label}>
                    <strong>{item.value}</strong>
                    <span>{item.label}</span>
                  </div>
                ))}
              </div>
              <div className="suggestion-grid">
                {suggestions.map((item) => (
                  <button key={item} onClick={() => askQuestion(item)}>{item}<span>↗</span></button>
                ))}
              </div>
              <div className="capability-grid">
                {capabilityCards.map((card) => (
                  <article className="capability-card" key={card.title}>
                    <strong>{card.title}</strong>
                    <span>{card.text}</span>
                  </article>
                ))}
              </div>
            </div>
          )}

          <div className="message-list">
            {messages.map((message, index) => (
              <div className={`message-row ${message.role}`} key={`${message.role}-${index}`}>
                <div className="message-avatar">{message.role === 'user' ? '李' : '文'}</div>
                <div className="message-body">
                  <div className="message-name">{message.role === 'user' ? '李老师' : '文脉助手'}</div>
                  <div className="message-content">{message.content}</div>
                  {message.citations?.length > 0 && (
                    <details className="citations">
                      <summary>查看 {message.citations.length} 条引用来源</summary>
                      <div className="citation-list">
                        {message.citations.map((citation) => (
                          <div className="citation" key={`${citation.chunk_id}-${citation.index}`}>
                            <span className="citation-number">{citation.index}</span>
                            <div><strong>{citation.document_type || '知识库资料'}</strong><span>{citation.section || citation.topic || citation.title || citation.source_file}</span></div>
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                </div>
              </div>
            ))}
            {loading && <div className="typing"><span /><span /><span /> 正在翻检案卷与记忆…</div>}
          </div>
        </section>

        <footer className="composer-wrap">
          {error && <div className="error-banner">{error}</div>}
          <div className="composer">
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  askQuestion()
                }
              }}
              placeholder="写下你的教学问题，文脉会记住最近对话…"
              rows="1"
            />
            <div className="composer-actions">
              <span>Enter 发送 · Redis 保存短期记忆</span>
              <button className="send-button" onClick={() => askQuestion()} disabled={loading || !question.trim()} aria-label="发送">↑</button>
            </div>
          </div>
          <p className="disclaimer">回答由知识库检索与对话记忆生成，请结合实际教学情况进行判断。</p>
        </footer>
      </main>
    </div>
  )
}

export default App
