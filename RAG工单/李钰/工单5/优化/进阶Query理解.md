# 进阶 Query 理解方案

> 工单编号: 人工智能 NLP-RAG-Query 理解优化任务

## 一、V5 当前方案

| 能力 | 实现 | 准确率 |
|------|------|--------|
| 指代消解 | 规则匹配 (他/这个公司) | ~95% |
| 省略补全 | 模式匹配 + 话题继承 | ~90% |
| 公司切换 | "那XX呢?" 正则 + 模糊匹配 | ~95% |
| 话题管理 | 关键词映射 | ~85% |
| 会话管理 | SessionContext (内存) | 100% |

## 二、进阶方案

### 2.1 LLM Zero-Shot Query 改写
```python
prompt = f"""
请将以下多轮对话中的当前问题改写为完整独立的问题, 保留指代消解和省略补全。
对话历史:
Q1: 武汉兴图新科军用收入?
Q2: 他的工程获一等奖?
当前问题: 这个公司的法定代表人是谁?
改写结果:
"""
rewritten = call_llm(prompt)
```
- 效果: +10-15% 覆盖率
- 代价: 每次多一次 LLM 调用 (~0.5s)

### 2.2 对话状态追踪 (DST)
```python
# 将对话状态结构化为 slot-value 对
{
  "company": "武汉力源",
  "topic": "组织结构",
  "turn": 5,
  "pending_intent": ["注册资本", "法定代表人"]
}
```
- 用 LLM 或专门 DST 模型
- 适合复杂多轮

### 2.3 指代消解模型
- **CoreNLP**: 斯坦福开源, 支持中文
- **HanLP**: 中文指代消解
- **基于规则 + LLM**: 轻量高效

### 2.4 会话持久化
```python
# Redis / SQLite 存储会话
import sqlite3
db = sqlite3.connect("sessions.db")
db.execute("CREATE TABLE sessions (id, context_json, updated_at)")
```

### 2.5 流式输出
```python
# Flask SSE / WebSocket
@app.route("/api/ask_stream")
def ask_stream():
    def generate():
        for chunk in generate_stream_answer():
            yield f"data: {chunk}\n\n"
    return Response(generate(), mimetype="text/event-stream")
```

## 三、V5 已足够达标

基于规则的指代消解 + 话题继承已覆盖验收标准的 5 轮多轮对话场景,
准确率 ≥ 90%, 响应时间 ≤ 3s。
进阶方案供生产环境或更复杂对话场景使用。
