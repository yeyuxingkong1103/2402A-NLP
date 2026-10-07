/**
 * 基于 PDF 文档的问答系统 - 前端逻辑（Query理解优化版 / 多轮对话）
 * 工单编号：人工智能NLP-RAG-Query理解优化任务
 * 功能：
 *   1. API 封装（fetch 调用后端 /api/upload, /api/ingest, /api/chat, /api/search）
 *   2. 多轮对话：维护 session_id，自动携带历史，支持指代消解和主语切换
 *   3. Query 改写展示：显示"理解后的问题"，让用户看见改写效果
 *   4. 问答交互（发送问题 → 流式接收回答 → 显示来源文档）
 *   5. 知识库管理（上传 PDF → 入库 → 显示状态）
 *   6. Markdown 渲染（marked.js CDN）
 *   7. Toast 提示
 *
 * 后端接口契约（见 src/main.py）：
 *   POST /api/upload   multipart 字段 file（单文件，前端逐个上传）
 *   POST /api/ingest   JSON {files?: [name]}   → {processed, ingested, collection_count}
 *   GET  /api/search?query=&top_k=             → {results:[{page_content,score,source,page_number}], total}
 *   POST /api/chat     JSON {query, top_k, stream, session_id, use_rewrite}
 *        - stream=true  → SSE：data: {"meta":true,...} / {"chunk":"..."} / {"done":true,"session_id":"..."} / {"error":"..."}
 *        - stream=false → {answer, rewritten_query, query_type, session_id, references:[...]}
 *   GET  /api/health   → {status, collection_count, ...}
 *   DELETE /api/session/{sid} → 清空当前会话
 */

// ==================== 配置 ====================
const API_BASE = '';   // 同源部署（前端由后端 /web 托管），留空即可

// ==================== 通用工具 ====================
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/** HTML 转义，避免把模型输出里的 < > 当标签解析 */
function escapeHtml(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

/** 把 Markdown 文本渲染成 HTML（失败则退化为转义纯文本） */
function renderMarkdown(md) {
    if (!md) return '';
    try {
        if (typeof marked !== 'undefined') {
            return marked.parse(md, { gfm: true, breaks: true });
        }
    } catch (e) {
        console.warn('marked 渲染失败，回退纯文本：', e);
    }
    return `<p>${escapeHtml(md).replace(/\n/g, '<br>')}</p>`;
}

/** Toast 提示
 * @param {string} msg   提示文本
 * @param {string} type  info | success | error | warn
 * @param {number} ms    自动消失时长（毫秒）
 */
function toast(msg, type = 'info', ms = 3000) {
    const box = $('#toasts');
    if (!box) return;
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = msg;
    box.appendChild(el);
    requestAnimationFrame(() => el.classList.add('show'));
    setTimeout(() => {
        el.classList.remove('show');
        setTimeout(() => el.remove(), 300);
    }, ms);
}

/** 连接状态指示灯 */
function setStatus(text, ok = false) {
    const dot = $('#statusDot');
    const txt = $('#statusText');
    if (dot) dot.className = `status-dot ${ok ? 'ok' : (text === '未连接' ? 'err' : 'warn')}`;
    if (txt) txt.textContent = text;
}

/** 把后端 references / results 的字段统一映射成前端用的格式 */
function normalizeSource(s) {
    return {
        content: s.page_content ?? s.content ?? '',
        source: s.source ?? '',
        score: typeof s.score === 'number' ? s.score : null,
        page: s.page_number ?? s.page ?? 0,
    };
}

// ==================== API 封装 ====================
const API = {

    /** 上传 PDF（后端单文件接口 file 字段，前端逐个上传）
     * @param {FileList|File[]} files
     * @returns {Promise<object[]>} 每个文件的服务器响应 {filename, saved_path, size}
     */
    async upload(files) {
        const out = [];
        for (const f of files) {
            if (!f.name.toLowerCase().endsWith('.pdf')) {
                toast(`跳过非 PDF 文件：${f.name}`, 'warn');
                continue;
            }
            const fd = new FormData();
            fd.append('file', f, f.name);
            const resp = await fetch(`${API_BASE}/api/upload`, {
                method: 'POST',
                body: fd,
            });
            if (!resp.ok) {
                const detail = await resp.text().catch(() => '');
                throw new Error(`上传 ${f.name} 失败 ${resp.status}：${detail}`);
            }
            out.push(await resp.json());
        }
        if (out.length === 0) throw new Error('未选择有效的 PDF 文件');
        return out;
    },

    /** 入库：解析 → 切分 → 向量化 → 写入 Milvus + 重建 BM25
     * @param {string[]} files 可选；为空/不传时处理 data 下所有 PDF
     * @returns {Promise<object>} {processed, ingested, collection_count}
     */
    async ingest(files = null) {
        const body = files && files.length ? { files } : {};
        const resp = await fetch(`${API_BASE}/api/ingest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!resp.ok) {
            const detail = await resp.text().catch(() => '');
            throw new Error(`入库失败 ${resp.status}：${detail}`);
        }
        return resp.json();
    },

    /** 检索（GET，不调 LLM）
     * @param {string} query
     * @param {number} topK
     * @returns {Promise<object[]>} 归一化后的来源数组
     */
    async search(query, topK = 5) {
        const url = `${API_BASE}/api/search?query=${encodeURIComponent(query)}&top_k=${topK}`;
        const resp = await fetch(url, { method: 'GET' });
        if (!resp.ok) {
            const detail = await resp.text().catch(() => '');
            throw new Error(`检索失败 ${resp.status}：${detail}`);
        }
        const data = await resp.json();
        return (data.results || []).map(normalizeSource);
    },

    /** 问答（流式接收回答）
     * 后端流式 SSE：data: {"chunk":"..."} / {"done":true} / {"error":"..."}
     * 流式不返回 references，所以调用方需另行调 /api/search 拿来源。
     * @param {string} query
     * @param {number} topK
     * @param {object}   handlers  { onToken, onDone, onError }
     * @param {AbortSignal} signal  可选，用于「停止」按钮中断
     */
    async chat(query, topK, handlers, signal, sessionId) {
        const { onToken = () => {}, onDone = () => {}, onError = () => {}, onMeta = () => {} } = handlers || {};
        let resp;
        try {
            resp = await fetch(`${API_BASE}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream, application/json' },
                body: JSON.stringify({
                    query,
                    top_k: topK,
                    stream: true,
                    session_id: sessionId || null,
                    use_rewrite: true,
                }),
                signal,
            });
        } catch (e) {
            if (e.name === 'AbortError') return;
            onError(e);
            return;
        }
        if (!resp.ok) {
            const detail = await resp.text().catch(() => '');
            onError(new Error(`问答失败 ${resp.status}：${detail}`));
            return;
        }

        const ct = resp.headers.get('content-type') || '';

        // —— 分支 1：非流式 JSON 兜底（后端忽略 stream=true 时）——
        if (ct.includes('application/json')) {
            const data = await resp.json();
            if (data.answer) onToken(data.answer);
            onDone(data);
            return;
        }

        // —— 分支 2：SSE 流式 ——
        const reader = resp.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buf = '';
        let fullAnswer = '';

        while (true) {
            let chunk;
            try {
                const r = await reader.read();
                if (r.done) break;
                chunk = r.value;
            } catch (e) {
                if (e.name === 'AbortError') break;
                onError(e);
                return;
            }
            buf += decoder.decode(chunk, { stream: true });

            const lines = buf.split('\n');
            buf = lines.pop();   // 最后一行可能不完整，留到下次
            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed || !trimmed.startsWith('data:')) continue;
                const payload = trimmed.replace(/^data:\s*/, '');
                let obj;
                try { obj = JSON.parse(payload); } catch { continue; }
                if (obj.meta) {
                    onMeta(obj);
                    continue;
                }
                if (obj.chunk) {
                    fullAnswer += obj.chunk;
                    onToken(obj.chunk);
                }
                if (obj.done) {
                    onDone({ answer: fullAnswer, session_id: obj.session_id });
                    return;
                }
                if (obj.error) {
                    onError(new Error(obj.error));
                    return;
                }
            }
        }
        onDone({ answer: fullAnswer });
    },

    /** 知识库健康检查（用作「文档/切片数」展示）
     * 后端 /api/health 返回 collection_count（切片数）；没有单独的文档数接口。
     */
    async stats() {
        try {
            const resp = await fetch(`${API_BASE}/api/health`, { method: 'GET' });
            if (!resp.ok) return null;
            return await resp.json();
        } catch {
            return null;
        }
    },
};

// ==================== 来源展示 ====================
const Sources = {
    /** 渲染检索来源列表
     * @param {Array<{content,source,score,page}>} srcs
     */
    render(srcs) {
        const list = $('#sourcesList');
        const count = $('#sourceCount');
        if (count) count.textContent = String(srcs ? srcs.length : 0);
        if (!list) return;
        if (!srcs || !srcs.length) {
            list.innerHTML = '<div class="sources-empty">回答后将在此显示引用的文档片段。</div>';
            return;
        }
        list.innerHTML = srcs.map((s, i) => {
            const src = s.source ? escapeHtml(s.source) : '未知来源';
            const score = (s.score !== null && typeof s.score === 'number') ? s.score.toFixed(4) : '-';
            const page = s.page ? ` · 第 ${escapeHtml(s.page)} 页` : '';
            const content = escapeHtml((s.content || '').slice(0, 400));
            return `
                <div class="source-item">
                    <div class="source-head">
                        <span class="source-idx">[${i + 1}]</span>
                        <span class="source-name">${src}${page}</span>
                        <span class="source-score">score: ${score}</span>
                    </div>
                    <div class="source-text">${content}</div>
                </div>`;
        }).join('');
    },
};

// ==================== 问答交互 ====================
const Chat = {
    sending: false,
    abortCtrl: null,
    // 多轮对话：session_id 持久化到 localStorage，刷新页面后继续保持会话
    sessionId: null,

    init() {
        const input = $('#questionInput');
        const sendBtn = $('#sendBtn');
        const newChatBtn = $('#newChatBtn');

        // 恢复 session_id
        try {
            this.sessionId = localStorage.getItem('rag_session_id') || null;
        } catch { this.sessionId = null; }

        input.addEventListener('input', () => this._autoGrow(input));
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.send();
            }
        });
        sendBtn.addEventListener('click', () => this.send());
        if (newChatBtn) newChatBtn.addEventListener('click', () => this.newSession());

        this._renderSessionBadge();
    },

    _renderSessionBadge() {
        const el = $('#sessionBadge');
        if (!el) return;
        if (this.sessionId) {
            el.textContent = `会话 ${this.sessionId.slice(0, 8)}…`;
            el.title = this.sessionId;
        } else {
            el.textContent = '新会话';
            el.title = '发送第一条消息后将自动创建会话';
        }
    },

    async newSession() {
        // 后端清空旧会话历史
        if (this.sessionId) {
            try {
                await fetch(`${API_BASE}/api/session/${encodeURIComponent(this.sessionId)}`, { method: 'DELETE' });
            } catch { /* ignore */ }
        }
        this.sessionId = null;
        try { localStorage.removeItem('rag_session_id'); } catch { }
        const history = $('#chatHistory');
        if (history) {
            history.innerHTML = `
                <div class="chat-empty" id="chatEmpty">
                  <div class="empty-icon">
                    <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                      <rect x="6" y="8" width="36" height="32" rx="4" stroke="currentColor" stroke-width="2.2"/>
                      <path d="M16 20h16M16 28h10" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
                    </svg>
                  </div>
                  <h2>已开始新会话</h2>
                  <p>历史已清空，继续提问将基于新的上下文。</p>
                </div>`;
        }
        Sources.render([]);
        this._renderSessionBadge();
        toast('已开始新会话', 'success');
    },

    _autoGrow(el) {
        el.style.height = 'auto';
        el.style.height = Math.min(el.scrollHeight, 200) + 'px';
    },

    async send() {
        if (this.sending) return;
        const input = $('#questionInput');
        const query = input.value.trim();
        if (!query) {
            toast('请输入问题', 'warn');
            return;
        }
        const empty = $('#chatEmpty');
        if (empty) empty.style.display = 'none';

        // 渲染用户气泡
        this._pushBubble('user', query);
        input.value = '';
        this._autoGrow(input);

        // 助手气泡占位（含改写展示区）
        const bubble = this._pushBubble('assistant', '');
        const contentEl = bubble.querySelector('.bubble-content');
        contentEl.innerHTML = '<span class="typing">正在检索并生成…</span>';

        // 先清空旧来源
        Sources.render([]);

        this._setSending(true);
        this.abortCtrl = new AbortController();

        // 检索来源：用原问题先查一次（流式 chat 不返回 references）
        // 实际检索是用改写后的问题在后端做的，这里仅做展示
        try {
            const srcs = await API.search(query, 5);
            Sources.render(srcs);
        } catch (e) {
            toast(`检索来源失败：${e.message}`, 'warn');
        }

        // —— 流式问答 ——
        let acc = '';
        let metaShown = false;
        await API.chat(query, 5, {
            onMeta: (meta) => {
                // 收到 meta：session_id + 改写后问题
                if (meta.session_id && !this.sessionId) {
                    this.sessionId = meta.session_id;
                    try { localStorage.setItem('rag_session_id', this.sessionId); } catch { }
                    this._renderSessionBadge();
                }
                // 如果发生了改写，显示"理解后的问题"
                if (meta.rewritten_query && meta.rewritten_query !== meta.original_query) {
                    const tag = {
                        'coreference': '指代消解',
                        'switch': '主语切换',
                        'followup': '主语继承',
                        'standalone': '独立问题',
                    }[meta.query_type] || meta.query_type;
                    const rewriteHtml = `
                        <div class="rewrite-info">
                            <span class="rewrite-tag">${escapeHtml(tag)}</span>
                            <span class="rewrite-label">理解后的问题：</span>
                            <span class="rewrite-text">${escapeHtml(meta.rewritten_query)}</span>
                        </div>`;
                    contentEl.innerHTML = rewriteHtml + '<div class="answer-body"><span class="typing">正在生成答案…</span></div>';
                    metaShown = true;
                }
            },
            onToken: (t) => {
                acc += t;
                const body = metaShown
                    ? contentEl.querySelector('.answer-body')
                    : contentEl;
                if (body) body.innerHTML = renderMarkdown(acc);
                this._scrollBottom();
            },
            onError: (e) => {
                contentEl.innerHTML = `<span class="err">回答失败：${escapeHtml(e.message)}</span>`;
                toast(`问答失败：${e.message}`, 'error', 5000);
            },
            onDone: (result) => {
                if (result && result.session_id && !this.sessionId) {
                    this.sessionId = result.session_id;
                    try { localStorage.setItem('rag_session_id', this.sessionId); } catch { }
                    this._renderSessionBadge();
                }
                if (!acc) {
                    const body = metaShown ? contentEl.querySelector('.answer-body') : contentEl;
                    if (body) body.textContent = '（未收到回答）';
                }
            },
        }, this.abortCtrl.signal, this.sessionId);

        this._setSending(false);
        this.abortCtrl = null;
    },

    /** 添加一个气泡到对话历史 */
    _pushBubble(role, text) {
        const history = $('#chatHistory');
        const empty = $('#chatEmpty');
        if (empty) empty.style.display = 'none';

        const wrap = document.createElement('div');
        wrap.className = `bubble-wrap bubble-${role}`;

        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.textContent = role === 'user' ? '我' : 'AI';

        const content = document.createElement('div');
        content.className = 'bubble-content';
        if (role === 'user') {
            content.textContent = text;
        } else if (text) {
            content.innerHTML = renderMarkdown(text);
        }

        wrap.appendChild(avatar);
        wrap.appendChild(content);
        history.appendChild(wrap);
        this._scrollBottom();
        return wrap;
    },

    _scrollBottom() {
        const history = $('#chatHistory');
        if (history) history.scrollTop = history.scrollHeight;
    },

    _setSending(sending) {
        this.sending = sending;
        const sendBtn = $('#sendBtn');
        const input = $('#questionInput');
        if (sending) {
            sendBtn.textContent = '停止';
            sendBtn.classList.add('stop');
            input.disabled = true;
        } else {
            sendBtn.textContent = '发送';
            sendBtn.classList.remove('stop');
            input.disabled = false;
            input.focus();
        }
    },
};

// ==================== 知识库管理 ====================
const KB = {
    pendingFiles: [],

    init() {
        const dropZone = $('#dropZone');
        const fileInput = $('#fileInput');
        const uploadBtn = $('#uploadBtn');
        const ingestBtn = $('#ingestBtn');
        const refreshBtn = $('#refreshBtn');

        dropZone.addEventListener('click', () => fileInput.click());
        ['dragenter', 'dragover'].forEach(ev =>
            dropZone.addEventListener(ev, e => {
                e.preventDefault();
                dropZone.classList.add('drag-over');
            }));
        ['dragleave', 'drop'].forEach(ev =>
            dropZone.addEventListener(ev, e => {
                e.preventDefault();
                dropZone.classList.remove('drag-over');
            }));
        dropZone.addEventListener('drop', e => {
            const files = e.dataTransfer.files;
            if (files && files.length) this._onPicked(files);
        });

        fileInput.addEventListener('change', e => {
            if (e.target.files && e.target.files.length) this._onPicked(e.target.files);
        });

        uploadBtn.addEventListener('click', () => this.upload());
        ingestBtn.addEventListener('click', () => this.ingest());
        refreshBtn.addEventListener('click', () => this.refresh());
    },

    _onPicked(files) {
        for (const f of files) {
            if (!f.name.toLowerCase().endsWith('.pdf')) {
                toast(`跳过非 PDF：${f.name}`, 'warn');
                continue;
            }
            if (this.pendingFiles.some(x => x.name === f.name && x.size === f.size)) continue;
            this.pendingFiles.push(f);
        }
        this._renderFileList();
        $('#uploadBtn').disabled = this.pendingFiles.length === 0;
    },

    _renderFileList() {
        const box = $('#fileList');
        if (!this.pendingFiles.length) {
            box.innerHTML = '<div class="file-empty">尚未选择任何 PDF</div>';
            return;
        }
        box.innerHTML = this.pendingFiles.map((f, i) => `
            <div class="file-item">
                <span class="file-name">${escapeHtml(f.name)}</span>
                <span class="file-size">${(f.size / 1024 / 1024).toFixed(2)} MB</span>
                <button class="file-del" data-idx="${i}" title="移除">×</button>
            </div>`).join('');
        $$('.file-del', box).forEach(btn => {
            btn.addEventListener('click', () => {
                const idx = +btn.dataset.idx;
                this.pendingFiles.splice(idx, 1);
                this._renderFileList();
                $('#uploadBtn').disabled = this.pendingFiles.length === 0;
            });
        });
    },

    async upload() {
        if (!this.pendingFiles.length) return;
        const btn = $('#uploadBtn');
        btn.disabled = true;
        btn.textContent = '上传中…';
        try {
            const resps = await API.upload(this.pendingFiles);
            toast(`已上传 ${resps.length} 个 PDF`, 'success');
            $('#ingestBtn').disabled = false;
            this.pendingFiles = [];
            this._renderFileList();
        } catch (e) {
            toast(e.message, 'error', 5000);
        } finally {
            btn.disabled = false;
            btn.textContent = '上传 PDF';
        }
    },

    async ingest() {
        const btn = $('#ingestBtn');
        btn.disabled = true;
        btn.textContent = '入库中…';
        try {
            const data = await API.ingest(null);
            toast(`入库完成：处理 ${data.processed} 个文件，成功 ${data.ingested} 个，集合总数 ${data.collection_count}`, 'success', 5000);
            await this.refresh();
        } catch (e) {
            toast(e.message, 'error', 5000);
        } finally {
            btn.disabled = false;
            btn.textContent = '入库';
        }
    },

    async refresh() {
        const docEl = $('#docCount');
        const chunkEl = $('#chunkCount');
        const status = $('#kbStatus');
        status.textContent = '正在获取知识库信息…';
        try {
            const data = await API.stats();
            if (!data) {
                docEl.textContent = '-';
                chunkEl.textContent = '-';
                status.textContent = '后端 /api/health 未响应。';
                return;
            }
            // 后端只返回集合切片数（collection_count），没有单独的文档数
            chunkEl.textContent = data.collection_count ?? '-';
            docEl.textContent = data.doc_count ?? '-';
            status.textContent = data.status === 'ok' ? '知识库就绪' : `状态：${data.status}`;
        } catch (e) {
            status.textContent = `获取失败：${e.message}`;
        }
    },
};

// ==================== 标签页切换 ====================
function initTabs() {
    $$('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.view;
            $$('.tab').forEach(t => t.classList.toggle('active', t === tab));
            $$('.view').forEach(v => v.classList.toggle('active', v.id === `view-${target}`));
            if (target === 'kb') KB.refresh();
        });
    });
}

// ==================== 后端探活 ====================
async function probeBackend() {
    try {
        const resp = await fetch(`${API_BASE}/api/health`, { method: 'GET' });
        if (resp.ok) {
            const data = await resp.json();
            setStatus(data.status === 'ok' ? '已连接' : '服务降级', data.status === 'ok');
            return;
        }
    } catch { /* ignore */ }
    setStatus('未连接', false);
}

// ==================== 启动 ====================
document.addEventListener('DOMContentLoaded', () => {
    if (typeof marked !== 'undefined') {
        marked.setOptions({ gfm: true, breaks: true });
    }
    initTabs();
    Chat.init();
    KB.init();
    KB._renderFileList();
    probeBackend();
});
