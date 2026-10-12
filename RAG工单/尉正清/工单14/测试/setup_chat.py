# 工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失
"""配置对话助手 —— 检索参数 + 系统提示词（本轮的全部「配置类」优化都在这）

对应工单里「调整配置」这条线，一共四件事：

1. **上重排**（rerank_id）。加了图纸描述以后，两条 2000+ 字的图注改变了 BM25 的
   语料统计，把带通用指令的查询向量吸了过去，纯文本题召回从 6 条掉到 1 条。
   RAGFlow 里重排发生在阈值过滤**之前**（rag/nlp/search.py:405-438），能把顺序纠回来。
2. **阈值降到 0.05**。重排分是 min-max 归一化的，交叉编码器打分又很尖
   （实测 0.995 / 0.018 / 1.6e-5），归一化后大部分候选接近 0，融合分
   `(1-w)·BM25 + w·重排分` 就压不过 0.2，等于把候选全滤掉。
3. **关掉 refine_multiturn**。开着的话每问会多一次 LLM 改写调用，白花 1 秒左右，
   而我们的问法最后一轮本来就是干净的题干，不需要改写。
4. **系统提示词**。规则要在知识库前后**各写一遍** —— 只写末尾会被大段知识淹没
   （实测第 5 题 5 次全错，前后各一次变 5 次全对）。

用法：
    ... setup_chat.py                    # 最终配置
    ... setup_chat.py --no-rerank        # 对比：不上重排
    ... setup_chat.py --default-prompt   # 对比：用 RAGFlow 默认提示词
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from ragflow_client import RAGFlow

HERE = Path(__file__).parent
IDS = json.loads((HERE / ".ragflow_ids.json").read_text(encoding="utf-8"))

# 助手的 rerank_id 校验用的是 tenant_llm 表的 llm_name 全名，**不带 @工厂** 后缀
# （api/apps/sdk/chat.py:84），写成 xxx@OpenAI-API-Compatible 会被判不存在
RERANK_ID = "bge-reranker-v2-m3___OpenAI-API"

RULES = """作答规则：
1. 题目给了选项时，只回答正确选项的字母（A、B、C 或 D），不要解释。
2. 涉及附图的题目，一律按图上箭头画出的实际指向判断气流/运动方向。
3. 「气流依次经过哪些部件」：按箭头指向排出零件次序后，剔除气流只是从
   内部流过的壳体（腔体、外壳、圆锥形部分），只保留气流真正**穿过**的
   零件（如配气带孔盘这类带孔件），再作答。"""

SYSTEM = f"""You are a precise assistant for Chinese industrial patent documents.
Answer strictly from the knowledge base below. If it does not contain the answer, say so.

{RULES}

Here is the knowledge base:
{{knowledge}}
The above is the knowledge base.

{RULES}"""

DEFAULT_SYSTEM = (
    "You are an intelligent assistant. Please summarize the content of the dataset to answer "
    "the question. Please list the data in the dataset and answer in detail. When all dataset "
    "content is irrelevant to the question, your answer must include the sentence \"The answer "
    "you are looking for is not found in the dataset!\" Answers need to consider chat history."
    "\n      Here is the knowledge base:\n      {knowledge}\n      The above is the knowledge base."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--thr", type=float, default=0.05, help="similarity_threshold")
    ap.add_argument("--vec", type=float, default=0.3, help="vector_similarity_weight")
    ap.add_argument("--topn", type=int, default=6)
    ap.add_argument("--refine", action="store_true", help="打开 refine_multiturn（默认关）")
    ap.add_argument("--default-prompt", action="store_true", help="用 RAGFlow 默认提示词")
    args = ap.parse_args()

    c = RAGFlow()
    c.load_token()
    body = {
        # dataset_ids 必须带上：SDK 的更新接口不带这个字段就会**把助手的知识库解绑**
        # （实测漏了它，kb_ids 直接变 []，之后走 async_chat_solo 分支，
        #  system prompt 里的 {knowledge} 原样发给模型，6 题全错）
        "dataset_ids": [IDS["ds_id"]],
        "rerank_id": "" if args.no_rerank else RERANK_ID,
        "similarity_threshold": args.thr,
        "vector_similarity_weight": args.vec,
        "top_n": args.topn,
        "prompt": {
            "prompt": DEFAULT_SYSTEM if args.default_prompt else SYSTEM,
            "refine_multiturn": bool(args.refine),
        },
    }
    code, j = c._req("PUT", f"/api/v1/chats/{IDS['chat_id']}", headers=c._h(), json=body)
    print(f"PUT HTTP {code} {json.dumps(j, ensure_ascii=False)[:160]}")

    code, j = c._req("GET", "/api/v1/chats", headers=c._h(),
                     params={"id": IDS["chat_id"]})
    d = (j.get("data") or [{}])[0]
    p = d.get("prompt")
    p = json.loads(p) if isinstance(p, str) else (p or {})
    print(f"  助手 {d.get('name')}")
    for k in ("rerank_model", "similarity_threshold", "keywords_similarity_weight",
              "top_n", "refine_multiturn"):
        print(f"    {k} = {p.get(k)}")


if __name__ == "__main__":
    main()
