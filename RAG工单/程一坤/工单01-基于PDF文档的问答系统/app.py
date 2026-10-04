# -*- coding: utf-8 -*-
"""
Web 问答界面（Flask）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
功能：简洁的聊天式 Web 界面，支持问题输入、答案与来源展示。
运行：python app.py  →  浏览器访问 http://127.0.0.1:8601
"""
import sys
import os
# 把上级目录的"00-公共模块"加入模块搜索路径，以便 import rag_engine / ollama_client
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00-公共模块"))

# Flask 核心：request 取请求参数、jsonify 返回 JSON、render_template_string 渲染内嵌页面模板
from flask import Flask, request, jsonify, render_template_string
# 导入 RAG 引擎：检索 + 生成 的完整问答流水线
from rag_engine import RAGEngine
# 导入 Ollama 客户端单例：启动前检测本地 Ollama 是否在线
from ollama_client import client

# 创建 Flask 应用实例
app = Flask(__name__)
engine = None  # 首次请求时懒加载

# 内嵌的聊天页面模板（HTML+CSS+JS）；{{ pre_html|safe }} 处由服务端注入预渲染的问答消息
PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>招股说明书问答系统 - 工单01</title>
<style>
  body{font-family:"Microsoft YaHei",sans-serif;background:#f4f6f9;margin:0;}
  .wrap{max-width:820px;margin:24px auto;padding:0 16px;}
  h1{font-size:20px;color:#1a3c6e;}
  .card{background:#fff;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.06);padding:20px;margin-bottom:16px;}
  .msg{padding:10px 14px;border-radius:8px;margin:8px 0;line-height:1.7;white-space:pre-wrap;}
  .user{background:#e3f0ff;align-self:flex-end;}
  .bot{background:#f1f3f5;}
  .meta{font-size:12px;color:#888;margin-top:4px;}
  .input-row{display:flex;gap:8px;}
  input[type=text]{flex:1;padding:10px 12px;border:1px solid #ccd5e0;border-radius:8px;font-size:15px;}
  button{padding:10px 22px;background:#1a5fb4;color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:15px;}
  button:disabled{background:#9db8d6;}
</style>
</head>
<body>
<div class="wrap">
  <h1>📖 招股说明书智能问答系统（工单01：基于PDF文档的问答系统）</h1>
  <div class="card" id="chat"><div class="msg bot">您好！我是招股说明书问答助手，请输入您的问题。</div>{{ pre_html|safe }}</div>
  <div class="card input-row">
    <input type="text" id="q" placeholder="例如：武汉兴图新科电子股份有限公司注册资本是多少？"
           onkeydown="if(event.key==='Enter')send()">
    <button id="btn" onclick="send()">提问</button>
  </div>
</div>
<script>
// 支持 URL 参数 ?q=问题 自动提问（便于演示与截图）
const urlQ=new URLSearchParams(location.search).get('q');
// send()：读取输入框问题→POST /api/ask→把用户消息与机器人回答插入聊天卡片
async function send(txt){
  const q=document.getElementById('q'); const btn=document.getElementById('btn');
  const text=q.value.trim(); if(!text) return;
  const chat=document.getElementById('chat');
  chat.insertAdjacentHTML('beforeend',`<div class="msg user">${text}</div>`);
  q.value=''; btn.disabled=true; btn.textContent='思考中…';
  const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:text})});
  const d=await r.json();
  chat.insertAdjacentHTML('beforeend',`<div class="msg bot">${d.answer||('出错: '+d.error)}
    <div class="meta">来源: ${d.sources||''} | 耗时: ${d.time_cost||''}s</div></div>`);
  btn.disabled=false; btn.textContent='提问'; chat.scrollTop=chat.scrollHeight;
}
// 页面加载 300ms 后若 URL 带 ?q= 则自动发送该问题（Chrome 无头截图依赖此逻辑）
if(urlQ) setTimeout(()=>send(urlQ), 300);
</script>
</body>
</html>"""


# 首页路由：GET / 时渲染聊天页面
@app.route("/")
def index():
    # 演示模式：URL 带 ?q=问题 时，服务端先完成问答并把结果预渲染进页面（便于截图演示）
    # 这样 Chrome 无头截图（不执行 JS 的等待场景）也能直接截到答案，而不是空聊天框
    q = request.args.get("q", "").strip()
    pre_html = ""  # 预渲染的问答 HTML 片段，默认为空（普通访问无预渲染内容）
    if q:
        from markupsafe import escape  # 防止演示参数注入 HTML
        global engine
        try:
            if engine is None:
                engine = RAGEngine("zgs1_v1")
            # 服务端同步执行完整 RAG 问答（检索+生成），确保 HTML 里已含答案
            r = engine.ask(q, with_context=True)
            # 把检索来源页码拼成"第x页, 第y页"格式的字符串
            src = ", ".join(f"第{s['page']}页" for s in r["sources"])
            # 组装预渲染片段：用户问题（经 escape 转义）+ 机器人答案 + 来源/耗时元信息
            pre_html = (
                f'<div class="msg user">{escape(q)}</div>'
                f'<div class="msg bot">{r["answer"]}'
                f'<div class="meta">来源: {src} | 耗时: {r["time_cost"]}s</div></div>'
            )
        except Exception as e:
            # 问答出错也预渲染出错误信息，保证截图/演示时页面有反馈
            pre_html = f'<div class="msg user">{escape(q)}</div><div class="msg bot">出错: {e}</div>'
    # 用内嵌模板 PAGE 渲染页面，pre_html 注入到聊天卡片中
    return render_template_string(PAGE, pre_html=pre_html)


# 问答接口：前端 JS 通过 POST /api/ask 发起异步提问
@app.route("/api/ask", methods=["POST"])
def ask():
    """问答接口：RAG 检索 + 生成"""
    global engine
    # 从 JSON 请求体取 question 字段；request.json 为空时兜底空 dict 防止抛异常
    q = (request.json or {}).get("question", "").strip()
    if not q:
        # 空问题直接返回错误提示，不进入检索流程
        return jsonify({"error": "问题不能为空"})
    try:
        if engine is None:
            engine = RAGEngine("zgs1_v1")
        # 执行完整 RAG 问答
        r = engine.ask(q, with_context=True)
        # 返回答案、来源页码串、总耗时给前端渲染
        return jsonify({
            "answer": r["answer"],
            "sources": ", ".join(f"第{s['page']}页" for s in r["sources"]),
            "time_cost": r["time_cost"],
        })
    except Exception as e:
        # 任何环节（embedding 检索 / LLM 生成）异常都包装为 JSON 错误返回
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    # 启动前最后检查 Ollama 服务，未启动则提示后不启动 Flask
    if not client.is_alive():
        print("[错误] Ollama 服务未启动")
    else:
        print("问答服务启动: http://127.0.0.1:8601")
        # 仅监听本机回环地址（演示安全），端口固定 8601，关闭 debug 避免自动重载
        app.run(host="127.0.0.1", port=8601, debug=False)
