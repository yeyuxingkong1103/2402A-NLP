# 测试夹具

## `qwen3_reasoning.txt` / `qwen3_answer.txt`

**真实模型输出**，不是手写的占位串。两个文件取自**同一次**生成：

- 模型：`qwen3:8b`（经 Ollama）
- 接口：原生 `/api/chat`，`stream=false`
- 问题：`高血压患者平时饮食和运动上要注意什么？`
- `qwen3_reasoning.txt` ← `message.thinking`（833 字，18 行）
- `qwen3_answer.txt` ← `message.content`（1670 字，73 行，含 `###` / `**` / `---`）

用来测 `app/core/postprocess.py` 的思维链清洗：把真实推理文本包进 think 标签，
断言「去 think 之后正文一字不改」。用真实文本而不是占位串，是为了覆盖中文口语、
换行、markdown 结构这些真实形态——手写夹具容易漏掉它们。

### 为什么要存文件而不是写在测试里

推理文本有 833 字，内联进测试文件会淹没断言；而且它是**外部数据**，
放在 `tests/fixtures/` 便于单独查看和更新。

### 怎么重新生成

模型升级（如换 `qwen3:14b`）或想换题目时，重跑这段即可：

```bash
python3 - <<'PY'
import json, urllib.request, pathlib
Q = "高血压患者平时饮食和运动上要注意什么？"
req = urllib.request.Request(
    "http://172.27.224.1:11434/api/chat",
    data=json.dumps({"model": "qwen3:8b",
                     "messages": [{"role": "user", "content": Q}],
                     "stream": False}).encode(),
    headers={"Content-Type": "application/json"},
)
m = json.loads(urllib.request.urlopen(req, timeout=900).read())["message"]
base = pathlib.Path("tests/fixtures")
(base / "qwen3_reasoning.txt").write_text(m["thinking"], encoding="utf-8")
(base / "qwen3_answer.txt").write_text(m["content"], encoding="utf-8")
PY
```

注意 `OLLAMA_BASE_URL` 里的主机地址是 WSL 访问宿主的地址，换机器要改。

### 一个容易误解的点

这些推理文本**在真实请求里不会进入 `postprocess`**——Ollama 把它放在独立字段，
`llm.py` 只取 `content`。夹具测的是「万一推理真出现在 content 里，清洗对不对」，
属于为其他 provider 准备的防御性覆盖。详见 `tests/test_think_stripping.py` 文件头。
