"""
基于 PDF 文档的问答系统 - 10 道工单测试问题对比测试
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

用法：
    conda activate rag_pdf_qa
    python tests/test_questions.py

作用：
    对工单要求的 10 个问题，分别跑两路回答：
      A. RAG 问答（调用后端 /api/chat，带知识库检索 + 重排 + 上下文拼装）
      B. 纯 LLM 问答（直接调 DeepSeek，不带任何检索）
    把两路回答 + 引用来源 + 耗时 + 是否命中关键词 一起写到：
      tests/results.json   机器可读
      tests/results.md     人工可读的对照表

依赖：
    httpx（已在 requirements.txt 里）
    环境变量 DEEPSEEK_API_KEY（从 .env 或 shell 读取）
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

# ============================================================
# 配置
# ============================================================

# 后端 FastAPI 地址（与 deploy/start.sh 中 API_PORT 一致）
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# DeepSeek 配置（纯 LLM 对照组用）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 检索 top_k
TOP_K = 5

# 超时（秒）：RAG 链路较慢，给宽松一点
TIMEOUT_RAG = 120.0
TIMEOUT_LLM = 60.0

# 结果输出路径
TESTS_DIR = Path(__file__).resolve().parent
RESULTS_JSON = TESTS_DIR / "results.json"
RESULTS_MD = TESTS_DIR / "results.md"


# ============================================================
# 10 道工单测试问题
# ============================================================
QUESTIONS = [
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？",
    "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？",
    "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？",
    "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？",
    "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？",
    "武汉兴图新科电子股份有限公司注册资本是多少？",
    "武汉兴图新科电子股份有限公司法定代表人是谁？",
    "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？",
]

# 每题的「期望关键词」：用于粗略判断回答是否命中要点（非严格评测，仅作参考）
# 这些词在标准答案中通常会出现；命中越多说明回答越靠谱
EXPECTED_KEYWORDS = {
    1: ["军用", "收入"],
    2: ["标准", "GB", "JT", "技术"],
    3: ["军用", "比重", "主营"],
    4: ["上游", "电子", "元器件", "芯片"],
    5: ["供应商", "重要"],
    6: ["下游", "行业", "应用"],
    7: ["工程", "国家科技进步", "一等奖"],
    8: ["注册资本", "万元"],
    9: ["法定代表人", "姓名"],
    10: ["募集资金", "补充流动资金", "%"],
}


# ============================================================
# 工具：加载 .env（兼容直接 python 运行，不依赖 python-dotenv）
# ============================================================
def load_dotenv():
    """从项目根 .env 文件加载环境变量（已存在的环境变量不覆盖）"""
    env_path = TESTS_DIR.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        # 只在环境变量未设置时才写入，shell 里 export 过的优先
        os.environ.setdefault(k, v)


# ============================================================
# A. RAG 问答：调用后端 /api/chat
# ============================================================
def call_rag(client: httpx.Client, question: str):
    """调用后端 /api/chat，返回 (answer, sources, elapsed, error)

    后端契约（src/main.py 的 ChatRequest）：
      请求：JSON {query, top_k, stream}   ← 字段名是 query 不是 question
      非流式响应：{query, answer, references: [{page_content, score, source, page_number}]}
      流式响应：SSE  data: {"chunk": "..."} / {"done": true} / {"error": "..."}
    本函数显式 stream=False 拿 JSON；若后端仍返回 SSE 则兜底解析。
    """
    t0 = time.time()
    try:
        resp = client.post(
            f"{API_BASE}/api/chat",
            json={"query": question, "top_k": TOP_K, "stream": False},
            timeout=TIMEOUT_RAG,
        )
    except Exception as e:
        return "", [], time.time() - t0, f"请求失败：{e}"

    if resp.status_code != 200:
        return "", [], time.time() - t0, f"HTTP {resp.status_code}: {resp.text[:300]}"

    ct = resp.headers.get("content-type", "")

    # —— 分支 1：非流式 JSON ——
    if "application/json" in ct:
        try:
            data = resp.json()
            return (
                data.get("answer", ""),
                data.get("references", []) or [],
                time.time() - t0,
                None,
            )
        except Exception as e:
            return "", [], time.time() - t0, f"JSON 解析失败：{e}"

    # —— 分支 2：SSE 流式（后端忽略 stream=False 时兜底）——
    answer_buf = []
    sources = []
    for line in resp.text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if "chunk" in obj:
            answer_buf.append(obj["chunk"])
        if "answer" in obj:
            answer_buf = [obj["answer"]]
        if obj.get("done"):
            break
        if "error" in obj:
            return "", [], time.time() - t0, f"后端错误：{obj['error']}"
        if isinstance(obj.get("references"), list) and obj["references"]:
            sources = obj["references"]
    return "".join(answer_buf), sources, time.time() - t0, None


# ============================================================
# B. 纯 LLM 问答：直接调 DeepSeek（不带任何检索）
# ============================================================
def call_llm(client: httpx.Client, question: str):
    """直接调用 DeepSeek chat completions，不经过知识库。

    返回 (answer, elapsed, error)
    用一个简单的系统提示词约束它基于常识回答（不编造招股书的具体数字）。
    """
    if not DEEPSEEK_API_KEY:
        return "", 0.0, "未配置 DEEPSEEK_API_KEY，跳过纯 LLM 对照"

    t0 = time.time()
    system = (
        "你是一个严谨的中文问答助手。请基于你已知的信息回答用户问题；"
        "如果不知道具体数字或事实，请明确说明「公开资料中未查询到该信息」，"
        "不要编造数据。回答尽量简洁。"
    )
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "temperature": 0.3,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        resp = client.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
            timeout=TIMEOUT_LLM,
        )
    except Exception as e:
        return "", time.time() - t0, f"请求失败：{e}"

    if resp.status_code != 200:
        return "", time.time() - t0, f"HTTP {resp.status_code}: {resp.text[:300]}"

    try:
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        return answer, time.time() - t0, None
    except Exception as e:
        return "", time.time() - t0, f"响应解析失败：{e}"


# ============================================================
# 关键词命中统计（粗略评测）
# ============================================================
def keyword_hit(answer: str, keywords):
    """返回 (命中数, 命中列表)；answer 为空时全 0"""
    if not answer:
        return 0, []
    hit = [kw for kw in keywords if kw in answer]
    return len(hit), hit


# ============================================================
# 主流程
# ============================================================
def main():
    load_dotenv()

    # 重新读一次环境变量（load_dotenv 之后）
    # 优先用大写名（DEEPSEEK_API_KEY），找不到则兼容后端 config.py 用的小写名
    # deepseek_api_key1 / deepseek_base_url，两套名在 .env 里都写了
    global DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek_api_key1", "")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("deepseek_base_url", DEEPSEEK_BASE_URL)
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL") or os.getenv("LLM_MODEL", DEEPSEEK_MODEL)

    print("=" * 60)
    print("  基于 PDF 文档的问答系统 - 10 道工单问题对比测试")
    print(f"  后端地址：{API_BASE}")
    print(f"  纯 LLM ：{DEEPSEEK_BASE_URL}  model={DEEPSEEK_MODEL}")
    print(f"  LLM 对照{'启用' if DEEPSEEK_API_KEY else '未启用（缺少 DEEPSEEK_API_KEY）'}")
    print("=" * 60)

    results = []
    with httpx.Client() as client:
        for idx, q in enumerate(QUESTIONS, 1):
            print(f"\n[{idx}/{len(QUESTIONS)}] {q}")

            # A. RAG
            rag_ans, rag_src, rag_t, rag_err = call_rag(client, q)
            if rag_err:
                print(f"  [RAG]  失败：{rag_err}")
            else:
                print(f"  [RAG]  {rag_t:.2f}s  来源数={len(rag_src)}  回答前 60 字：{(rag_ans or '')[:60]}...")

            # B. 纯 LLM
            llm_ans, llm_t, llm_err = call_llm(client, q)
            if llm_err:
                print(f"  [LLM]  跳过：{llm_err}")
            else:
                print(f"  [LLM]  {llm_t:.2f}s  回答前 60 字：{(llm_ans or '')[:60]}...")

            # 关键词命中
            kws = EXPECTED_KEYWORDS.get(idx, [])
            rag_hit_n, rag_hit_list = keyword_hit(rag_ans, kws)
            llm_hit_n, llm_hit_list = keyword_hit(llm_ans, kws)

            results.append({
                "idx": idx,
                "question": q,
                "expected_keywords": kws,
                "rag": {
                    "answer": rag_ans,
                    "sources": rag_src,
                    "elapsed_sec": round(rag_t, 2),
                    "error": rag_err,
                    "keyword_hit": rag_hit_n,
                    "keyword_hit_list": rag_hit_list,
                    "keyword_total": len(kws),
                },
                "llm": {
                    "answer": llm_ans,
                    "elapsed_sec": round(llm_t, 2),
                    "error": llm_err,
                    "keyword_hit": llm_hit_n,
                    "keyword_hit_list": llm_hit_list,
                    "keyword_total": len(kws),
                },
            })

    # ---------- 写 results.json ----------
    RESULTS_JSON.write_text(
        json.dumps({"api_base": API_BASE, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n已写出：{RESULTS_JSON}")

    # ---------- 写 results.md ----------
    md = render_md(results)
    RESULTS_MD.write_text(md, encoding="utf-8")
    print(f"已写出：{RESULTS_MD}")

    # ---------- 汇总 ----------
    rag_ok = sum(1 for r in results if not r["rag"]["error"])
    llm_ok = sum(1 for r in results if not r["llm"]["error"])
    rag_kw_total = sum(r["rag"]["keyword_total"] for r in results)
    rag_kw_hit = sum(r["rag"]["keyword_hit"] for r in results)
    llm_kw_total = sum(r["llm"]["keyword_total"] for r in results)
    llm_kw_hit = sum(r["llm"]["keyword_hit"] for r in results)

    print("\n" + "=" * 60)
    print(f"  RAG 成功 {rag_ok}/{len(QUESTIONS)}，关键词命中 {rag_kw_hit}/{rag_kw_total}")
    print(f"  LLM 成功 {llm_ok}/{len(QUESTIONS)}，关键词命中 {llm_kw_hit}/{llm_kw_total}")
    print("=" * 60)


# ============================================================
# Markdown 报告渲染
# ============================================================
def render_md(results):
    lines = []
    lines.append("# 基于 PDF 文档的问答系统 - 10 道工单问题对比测试结果\n")
    lines.append(f"- 后端地址：`{API_BASE}`")
    lines.append(f"- 纯 LLM：`{DEEPSEEK_BASE_URL}`  model=`{DEEPSEEK_MODEL}`")
    lines.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## 汇总\n")
    lines.append("| # | 问题（前 30 字） | RAG 耗时(s) | RAG 关键词命中 | LLM 耗时(s) | LLM 关键词命中 |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        q = r["question"][:30] + ("…" if len(r["question"]) > 30 else "")
        rag_t = r["rag"]["elapsed_sec"] if not r["rag"]["error"] else "ERR"
        rag_kw = f"{r['rag']['keyword_hit']}/{r['rag']['keyword_total']}" if not r["rag"]["error"] else "-"
        llm_t = r["llm"]["elapsed_sec"] if not r["llm"]["error"] else "SKIP"
        llm_kw = f"{r['llm']['keyword_hit']}/{r['llm']['keyword_total']}" if not r["llm"]["error"] else "-"
        lines.append(f"| {r['idx']} | {q} | {rag_t} | {rag_kw} | {llm_t} | {llm_kw} |")
    lines.append("")

    lines.append("## 详细对比\n")
    for r in results:
        lines.append(f"### Q{r['idx']}：{r['question']}\n")
        lines.append(f"**期望关键词**：{', '.join(r['expected_keywords']) or '（无）'}\n")

        lines.append("#### A. RAG 问答（带检索）\n")
        if r["rag"]["error"]:
            lines.append(f"- ❌ 错误：`{r['rag']['error']}`\n")
        else:
            lines.append(f"- ⏱ 耗时：{r['rag']['elapsed_sec']} s")
            lines.append(f"- 🔑 关键词命中：{r['rag']['keyword_hit']}/{r['rag']['keyword_total']}（{', '.join(r['rag']['keyword_hit_list']) or '无'}）")
            lines.append(f"- 📄 引用来源数：{len(r['rag']['sources'])}\n")
            lines.append("```text")
            lines.append(r["rag"]["answer"] or "（空）")
            lines.append("```\n")
            if r["rag"]["sources"]:
                lines.append("**检索来源**：\n")
                for i, s in enumerate(r["rag"]["sources"], 1):
                    src = s.get("source", "未知") if isinstance(s, dict) else str(s)
                    score = s.get("score", "-") if isinstance(s, dict) else "-"
                    # 后端 references 用 page_content，search 用 page_content，兼容旧 content 字段
                    content = (s.get("page_content") or s.get("content", "") if isinstance(s, dict) else str(s))[:200]
                    lines.append(f"{i}. `{src}`  score={score}")
                    lines.append(f"   > {content}...")
                lines.append("")

        lines.append("#### B. 纯 LLM 问答（无检索）\n")
        if r["llm"]["error"]:
            lines.append(f"- ⚠ 跳过：`{r['llm']['error']}`\n")
        else:
            lines.append(f"- ⏱ 耗时：{r['llm']['elapsed_sec']} s")
            lines.append(f"- 🔑 关键词命中：{r['llm']['keyword_hit']}/{r['llm']['keyword_total']}（{', '.join(r['llm']['keyword_hit_list']) or '无'}）\n")
            lines.append("```text")
            lines.append(r["llm"]["answer"] or "（空）")
            lines.append("```\n")

        lines.append("---\n")

    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
