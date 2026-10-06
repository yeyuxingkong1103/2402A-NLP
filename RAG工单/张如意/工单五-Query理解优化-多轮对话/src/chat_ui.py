# -*- coding: utf-8 -*-
"""
工单05 带界面的多轮对话（交互友好性验收项）
工单编号：人工智能NLP-RAG-Query理解优化任务

界面能力（对应验收标准「交互友好性」）：
  1. 对话历史展示：左侧气泡式聊天记录，支持连续多轮追问
  2. 指代消解过程可视化：右侧面板展示每轮「原始问题 → 指代消解后的检索式」，
     并给出识别到的意图 / 实体 / 子问题，让用户看到系统「理解了什么」
  3. 引用来源：每条答案下方列出《文档》页码与片段摘要
  4. 用户反馈：👍 有帮助 / 👎 没帮助 按钮，反馈写回
     rag_core.rerank.AdaptiveReranker（在线学习，越用越准）
  5. 示例问题一键填充：内置 config.MULTI_TURN_SCRIPT 五轮脚本
  6. 多语言：中文提问中文答、英文提问英文答（由 rag_core.generator 判定）
  7. 容错：检索/生成异常时在气泡中提示错误原因，会话不中断

两种运行模式（自动选择）：
  · 已安装 Gradio：启动完整 Gradio 界面
        pip install gradio        # 未安装时给出提示
  · 未安装 Gradio：自动降级为内置标准库 HTTP 界面（零额外依赖），
        功能一致（多轮、改写可视化、引用、反馈）

运行：
    python chat_ui.py                 # 自动选择 UI 后端
    python chat_ui.py --port 7860
    python chat_ui.py --mode http     # 强制使用内置 HTTP 界面
    python chat_ui.py --mode gradio   # 强制 Gradio（未安装则报错提示）
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_core import config                                        # noqa: E402
from rag_core.pipeline import Pipeline                             # noqa: E402
from rag_core.query_understand import Turn                         # noqa: E402
from rag_core.rerank import AdaptiveReranker                       # noqa: E402

from wo05_common import (                                          # noqa: E402
    COLLECTION, build_pipeline, index_ready,
)

UI_VERSION = "工单05 多轮对话演示 v1.0"


# ---------------------------------------------------------------------------
# 会话对象：界面与后端之间唯一的耦合点
# ---------------------------------------------------------------------------
class ChatSession:
    """
    一次浏览器会话：维护对话历史、Query 理解痕迹、检索片段与用户反馈。

    说明：界面上展示的「改写过程」直接来自 rag_core.query_understand 的输出，
    不做二次加工，保证「所见即系统真实行为」。
    """

    def __init__(self, pipeline: Pipeline, top_k: int = 5):
        self.pipeline = pipeline
        self.top_k = top_k
        self.history: list[Turn] = []
        self.turns: list[dict] = []               # 界面用的逐轮展示数据

    # -- 问答 ---------------------------------------------------------------
    def ask(self, question: str) -> dict:
        """执行一轮问答，返回界面需要的全部字段（带容错）。"""
        t0 = time.perf_counter()
        if not question or not question.strip():
            return {"ok": False, "error": "请输入问题", "answer": ""}

        try:
            trace = self.pipeline.ask(question, history=self.history,
                                      top_k=self.top_k, return_trace=True)
        except Exception as e:                    # 容错：界面提示而不是崩溃
            msg = f"处理失败：{type(e).__name__}: {e}"
            self.history.append(Turn(question=question, answer=""))
            turn = {"question": question, "answer": msg, "error": str(e),
                    "rewritten": question, "intent": "", "entities": [],
                    "sub_questions": [], "docs": [], "latency": time.perf_counter() - t0}
            self.turns.append(turn)
            return {"ok": False, "error": msg, "answer": msg, **turn}

        qu = trace.get("understanding", {})
        docs = trace.get("docs") or []
        turn = {
            "question": question,
            "answer": trace.get("answer", ""),
            "rewritten": qu.get("改写后", question),
            "intent": qu.get("意图", ""),
            "entities": qu.get("实体", []),
            "sub_questions": qu.get("子问题", []),
            "ambiguity": qu.get("歧义提示"),
            "citations": trace.get("citations", []),
            "docs": [
                {"doc": d.get("doc"), "page": d.get("page"), "type": d.get("type"),
                 "section": d.get("section"),
                 "score": round(float(d.get("final_score", d.get("score", 0.0))), 4),
                 "text": (d.get("text") or "")[:300]}
                for d in docs[:3]
            ],
            "timings": {k: round(v, 3) for k, v in (trace.get("timings") or {}).items()},
            "latency": round(time.perf_counter() - t0, 3),
            "error": None,
        }
        self.turns.append(turn)
        self.history.append(Turn(question=question, answer=turn["answer"], docs=docs))
        return {"ok": True, **turn}

    # -- 反馈 ---------------------------------------------------------------
    def feedback(self, turn_no: int, helpful: bool) -> dict:
        """
        记录用户反馈并写回自适应重排器。
        turn_no 为界面上的轮次序号（1 起）；对检索 Top-3 片段统一打标。
        """
        if not (1 <= turn_no <= len(self.turns)):
            return {"ok": False, "msg": "无效的轮次"}
        turn = self.turns[turn_no - 1]
        rr = AdaptiveReranker()                   # 内部自动 load/save 反馈文件
        docs = turn.get("docs") or []
        if not docs:
            docs = [{"chunk_id": "", "text": turn.get("answer", ""), "type": "text"}]
        for d in docs:
            rr.record(turn["question"], d, helpful)
        # 反馈事件也记到轮次上，便于演示时展示
        turn.setdefault("feedback", []).append("有帮助" if helpful else "没帮助")
        return {"ok": True, "msg": "感谢反馈，已写入自适应重排器（下次相似问题会调整排序）",
                "stats": rr.stats()}

    def reset(self) -> None:
        self.history.clear()
        self.turns.clear()

    # -- 展示 ---------------------------------------------------------------
    def rewrite_panel(self) -> str:
        """指代消解过程可视化（Markdown）。"""
        if not self.turns:
            return ("### 指代消解过程\n\n"
                    "提问后这里会显示：**原始问题 → 指代消解后的检索式**，"
                    "以及识别到的意图、实体、子问题。\n\n"
                    "试试第 4 轮：「那武汉力源信息技术股份有限公司呢？」")
        lines = ["### 指代消解过程", ""]
        for i, t in enumerate(self.turns, 1):
            changed = t["rewritten"] != t["question"]
            lines.append(f"**第 {i} 轮**")
            lines.append(f"- 原始问题：{t['question']}")
            if changed:
                lines.append(f"- 检索式（已改写）：**{t['rewritten']}**")
            else:
                lines.append("- 检索式：与原始问题一致（无需改写）")
            lines.append(f"- 意图：{t.get('intent') or '（未识别）'}"
                         f"　实体：{'、'.join(t.get('entities') or []) or '（无）'}")
            if len(t.get("sub_questions") or []) > 1:
                lines.append(f"- 子问题：{'；'.join(t['sub_questions'])}")
            if t.get("ambiguity"):
                lines.append(f"- 消歧提示：{t['ambiguity']}")
            if t.get("error"):
                lines.append(f"- 异常：{t['error']}")
            lines.append("")
        return "\n".join(lines)

    def citations_md(self) -> str:
        """引用来源 + 检索片段（Markdown）。"""
        if not self.turns:
            return "### 引用来源\n\n（暂无）"
        t = self.turns[-1]
        lines = ["### 引用来源（最近一轮）", ""]
        for d in t.get("docs") or []:
            snippet = (d.get("text") or "").replace("\n", " ")[:100]
            lines.append(f"- 《{d.get('doc')}》第 {d.get('page')} 页"
                         f"（{d.get('type')}，相关度 {d.get('score')}）：{snippet}…")
        if not (t.get("docs") or []):
            lines.append("（无检索片段）")
        return "\n".join(lines)

    def latency_md(self) -> str:
        """耗时与 3 秒约束自检。"""
        if not self.turns:
            return "### 性能\n\n（暂无）"
        t = self.turns[-1]
        timings = "；".join(f"{k} {v}s" for k, v in (t.get("timings") or {}).items())
        ok = "满足" if t.get("latency", 9) <= 3 else "超出"
        return (f"### 性能\n\n本轮耗时 **{t.get('latency', 0)}s**"
                f"（3 秒约束：{ok}）\n\n阶段明细：{timings or '—'}")


# ---------------------------------------------------------------------------
# 会话管理（按浏览器会话 id 隔离）
# ---------------------------------------------------------------------------
_SESSIONS: dict[str, ChatSession] = {}
_PIPELINE: Pipeline | None = None
_LOCK = threading.Lock()


def get_pipeline() -> Pipeline:
    """惰性初始化流水线（首次请求时装载索引）。"""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = build_pipeline(use_query_understanding=True, collection=COLLECTION)
    return _PIPELINE


def get_session(sid: str, top_k: int = 5) -> ChatSession:
    with _LOCK:
        if sid not in _SESSIONS:
            _SESSIONS[sid] = ChatSession(get_pipeline(), top_k=top_k)
        return _SESSIONS[sid]


# ---------------------------------------------------------------------------
# Gradio 界面
# ---------------------------------------------------------------------------
def launch_gradio(port: int, top_k: int) -> int:
    """启动 Gradio 界面（需 pip install gradio）。"""
    import gradio as gr

    major = int(str(gr.__version__).split(".")[0])
    msg_mode = major >= 5                     # Gradio 5+ 使用 messages 格式

    def _display(records: list[dict]) -> list:
        """把内部轮次数据转成 Chatbot 组件需要的格式。"""
        out = []
        for t in records:
            ans = t["answer"] or "（无答案）"
            cites = t.get("citations") or []
            if cites:
                ans += "\n\n来源：" + "、".join(
                    f"《{c.get('doc')}》第{c.get('page')}页" for c in cites[:3])
            ans += (f"\n\n（本轮耗时 {t.get('latency')}s"
                    f"{'，已改写检索式' if t['rewritten'] != t['question'] else ''}）")
            if msg_mode:
                out.append({"role": "user", "content": t["question"]})
                out.append({"role": "assistant", "content": ans})
            else:
                out.append((t["question"], ans))
        return out

    def ui_ask(message, display, sid):
        sess = get_session(sid or "default", top_k)
        if not (message or "").strip():
            return display, message, sess.rewrite_panel(), sess.citations_md(), \
                gr.update(), sess.latency_md()
        sess.ask(message)
        return (_display(sess.turns), "", sess.rewrite_panel(), sess.citations_md(),
                gr.update(choices=[f"第 {i} 轮" for i in range(1, len(sess.turns) + 1)],
                          value=f"第 {len(sess.turns)} 轮"),
                sess.latency_md())

    def ui_feedback(choice, helpful, sid):
        sess = get_session(sid or "default", top_k)
        try:
            no = int(str(choice).replace("第", "").replace("轮", "").strip())
        except Exception:
            no = len(sess.turns)
        r = sess.feedback(no, helpful)
        stats = r.get("stats", {})
        return f"{r.get('msg', '')}　当前反馈统计：{stats}"

    def ui_reset(sid):
        sess = get_session(sid or "default", top_k)
        sess.reset()
        return [], "", sess.rewrite_panel(), sess.citations_md(), \
            gr.update(choices=[], value=None), sess.latency_md()

    with gr.Blocks(title="工单05 多轮对话演示", theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# 招股说明书多轮问答（{UI_VERSION}）\n"
                    "支持指代消解：「他」「这个公司」「那 X 呢？」会自动改写为"
                    "可独立检索的完整问题，并展示改写过程。")
        # 每个浏览器会话一个独立 session id，对话历史互不干扰
        sid = gr.State(lambda: uuid.uuid4().hex[:8])
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = (gr.Chatbot(label="对话历史", height=460, type="messages")
                           if msg_mode else gr.Chatbot(label="对话历史", height=460))
                with gr.Row():
                    msg = gr.Textbox(placeholder="请输入问题，回车发送…",
                                     label="", scale=5, autofocus=True)
                    send = gr.Button("发送", variant="primary", scale=1)
                with gr.Row():
                    reset = gr.Button("清空对话")
                    gr.Examples(
                        examples=[[q] for q in config.MULTI_TURN_SCRIPT],
                        inputs=[msg], label="五轮示例（点击填充）")
            with gr.Column(scale=2):
                panel = gr.Markdown("### 指代消解过程\n\n提问后显示改写过程。")
                cites = gr.Markdown("### 引用来源\n\n（暂无）")
                perf = gr.Markdown("### 性能\n\n（暂无）")
                fb_no = gr.Dropdown(label="反馈轮次", choices=[], value=None)
                with gr.Row():
                    up = gr.Button("有帮助 👍")
                    down = gr.Button("没帮助 👎")
                fb_status = gr.Markdown("反馈将写回自适应重排器（AdaptiveReranker）")

        outputs = [chatbot, msg, panel, cites, fb_no, perf]
        send.click(ui_ask, [msg, chatbot, sid], outputs)
        msg.submit(ui_ask, [msg, chatbot, sid], outputs)
        reset.click(ui_reset, [sid], outputs)
        up.click(lambda c, s: ui_feedback(c, True, s), [fb_no, sid], [fb_status])
        down.click(lambda c, s: ui_feedback(c, False, s), [fb_no, sid], [fb_status])

    demo.launch(server_port=port, show_error=True)
    return 0


# ---------------------------------------------------------------------------
# 内置 HTTP 界面（零依赖降级方案）
# ---------------------------------------------------------------------------
_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>工单05 多轮对话演示</title>
<style>
 body{margin:0;font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#f5f7fa;color:#1f2937}
 header{background:#1e3a5f;color:#fff;padding:12px 22px}
 header h1{font-size:16px;margin:0}
 .wrap{display:flex;gap:14px;padding:14px;height:calc(100vh - 92px)}
 .left{flex:3;display:flex;flex-direction:column;background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
 #chat{flex:1;overflow-y:auto;padding:16px}
 .msg{margin-bottom:14px;display:flex;gap:8px}
 .msg.user{flex-direction:row-reverse}
 .bubble{max-width:82%;padding:10px 14px;border-radius:10px;background:#f3f4f6;white-space:pre-wrap;line-height:1.7;font-size:14px}
 .user .bubble{background:#dbeafe}
 .cite{font-size:12px;color:#6b7280;margin-top:6px}
 .fb{font-size:12px;margin-top:6px}
 .fb button{border:1px solid #d1d5db;background:#fff;border-radius:5px;cursor:pointer;margin-right:6px}
 .bar{display:flex;gap:8px;padding:12px;border-top:1px solid #e5e7eb}
 input[type=text]{flex:1;padding:10px;border:1px solid #d1d5db;border-radius:8px;font-size:14px}
 button.send{background:#2563eb;color:#fff;border:none;border-radius:8px;padding:10px 20px;cursor:pointer}
 .right{flex:2;display:flex;flex-direction:column;gap:12px;overflow-y:auto}
 .card{background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:14px;font-size:13px;line-height:1.7}
 .card h3{margin:0 0 8px;font-size:14px;color:#1e3a5f}
 .rw{background:#fff7ed;border-left:3px solid #f59e0b;padding:6px 8px;border-radius:4px;margin:6px 0}
 .hint{font-size:12px;color:#9ca3af}
</style></head><body>
<header><h1>招股说明书多轮问答 · 工单05（指代消解可视化）</h1></header>
<div class="wrap">
  <div class="left">
    <div id="chat"></div>
    <div class="bar">
      <input type="text" id="q" placeholder="请输入问题，回车发送…" autocomplete="off">
      <button class="send" id="send">发送</button>
      <button class="send" style="background:#6b7280" id="reset">清空</button>
    </div>
  </div>
  <div class="right">
    <div class="card" id="panel"><h3>指代消解过程</h3><span class="hint">提问后显示「原始问题 → 改写后检索式」。</span></div>
    <div class="card" id="cites"><h3>引用来源</h3><span class="hint">（暂无）</span></div>
    <div class="card" id="perf"><h3>性能</h3><span class="hint">（暂无）</span></div>
    <div class="card" id="fbcard"><h3>反馈</h3><span class="hint">对最近一轮点赞/点踩，写回自适应重排器。</span></div>
  </div>
</div>
<script>
const chat=document.getElementById('chat'),q=document.getElementById('q');
let turns=[];
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function render(){
  chat.innerHTML='';
  turns.forEach((t,i)=>{
    const u=document.createElement('div');u.className='msg user';
    u.innerHTML='<div class="bubble">'+esc(t.question)+'</div>';chat.appendChild(u);
    const b=document.createElement('div');b.className='msg';
    let cites=(t.citations||[]).map(c=>'《'+esc(c.doc)+'》第'+esc(c.page)+'页').join('、');
    b.innerHTML='<div class="bubble">'+esc(t.answer)+
      (cites?'<div class="cite">来源：'+cites+'</div>':'')+
      '<div class="cite">本轮耗时 '+t.latency+'s'+(t.rewritten!==t.question?' · 已改写检索式':'')+'</div>'+
      '<div class="fb"><button data-i="'+i+'" data-v="1">有帮助</button>'+
      '<button data-i="'+i+'" data-v="0">没帮助</button></div></div>';
    chat.appendChild(b);
  });
  chat.scrollTop=chat.scrollHeight;
  document.querySelectorAll('.fb button').forEach(b=>b.onclick=()=>fb(+b.dataset.i,+b.dataset.v));
  const p=turns.length?turns[turns.length-1]:null;
  document.getElementById('panel').innerHTML='<h3>指代消解过程</h3>'+(p?
    ('<div>原始问题：'+esc(p.question)+'</div>'+
     (p.rewritten!==p.question?'<div class="rw">检索式（已改写）：<b>'+esc(p.rewritten)+'</b></div>':'<div>检索式：与原始问题一致（无需改写）</div>')+
     '<div>意图：'+esc(p.intent)+'　实体：'+esc((p.entities||[]).join('、'))+'</div>'+
     ((p.sub_questions||[]).length>1?'<div>子问题：'+esc(p.sub_questions.join('；'))+'</div>':'')):'<span class="hint">（暂无）</span>');
  document.getElementById('cites').innerHTML='<h3>引用来源</h3>'+(p&&(p.docs||[]).length?
    p.docs.map(d=>'<div>《'+esc(d.doc)+'》第 '+esc(d.page)+' 页（'+esc(d.type)+'，相关度 '+d.score+'）</div>').join(''):'<span class="hint">（暂无）</span>');
  document.getElementById('perf').innerHTML='<h3>性能</h3>'+(p?
    ('本轮耗时 <b>'+p.latency+'s</b>（3 秒约束：'+(p.latency<=3?'满足':'超出')+'）<br>阶段：'+esc(JSON.stringify(p.timings))):'<span class="hint">（暂无）</span>');
}
async function ask(){
  const text=q.value.trim(); if(!text) return; q.value='';
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:text})});
    const d=await r.json(); turns.push(d); render();
  }catch(e){alert('请求失败：'+e);}
}
async function fb(i,v){
  const r=await fetch('/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({turn:i+1,helpful:!!v})});
  const d=await r.json();
  document.getElementById('fbcard').innerHTML='<h3>反馈</h3>'+esc(d.msg||'')+'<br><span class="hint">'+esc(JSON.stringify(d.stats||{}))+'</span>';
}
document.getElementById('send').onclick=ask;
q.addEventListener('keydown',e=>{if(e.key==='Enter')ask();});
document.getElementById('reset').onclick=async()=>{await fetch('/reset',{method:'POST'});turns=[];render();document.getElementById('fbcard').innerHTML='<h3>反馈</h3><span class="hint">已清空会话</span>';};
q.focus();
</script></body></html>"""


def launch_http(port: int, top_k: int) -> int:
    """内置标准库 HTTP 界面（无 Gradio 依赖时的降级方案，单会话演示模式）。"""
    session = ChatSession(get_pipeline(), top_k=top_k)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self):                                   # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(200, _HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/health":
                self._json({"status": "ok", "ui": "http", "version": UI_VERSION})
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")

        def do_POST(self):                                  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                payload = {}
            if self.path == "/ask":
                self._json(session.ask(payload.get("question", "")))
            elif self.path == "/feedback":
                self._json(session.feedback(int(payload.get("turn", 0)),
                                            bool(payload.get("helpful"))))
            elif self.path == "/reset":
                session.reset()
                self._json({"ok": True})
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")

        def log_message(self, fmt, *args):                  # 静默访问日志
            pass

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"内置 HTTP 界面已启动：http://127.0.0.1:{port}/")
    print("提示：安装 Gradio 可获得更完整的界面：pip install gradio")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="工单05 多轮对话界面")
    ap.add_argument("--port", type=int, default=7860, help="端口（默认 7860）")
    ap.add_argument("--top-k", type=int, default=5, help="送入生成的片段数")
    ap.add_argument("--mode", choices=["auto", "gradio", "http"], default="auto",
                    help="UI 后端；auto=有 Gradio 用 Gradio，否则降级 HTTP")
    args = ap.parse_args()

    if not index_ready(COLLECTION):
        print("[提示] 未检测到索引，请先运行：python build_index.py")
        return 2

    if args.mode in ("auto", "gradio"):
        try:
            import gradio  # noqa: F401
            return launch_gradio(args.port, args.top_k)
        except ImportError:
            if args.mode == "gradio":
                print("[错误] 未安装 Gradio。请执行：pip install gradio")
                return 1
            print("[提示] 未检测到 Gradio，降级使用内置 HTTP 界面。")
    return launch_http(args.port, args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
