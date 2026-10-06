# 工单编号：人工智能 NLP-RAG-Graph RAG 优化任务
import json
import re
import networkx as nx
from openai import OpenAI
from config_v9 import CHUNK_FILE, GRAPH_FILE, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
import os

llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

PROMPT = """从下面的文本中抽取实体和关系，输出 JSON。

要求：
1. 实体类型：公司、机构、人物、指标、年份、业务、地区
2. 关系类型：控股、任职、披露、同比、包含、位于
3. 每条 chunk 最多抽 5 个实体、3 条关系
4. 只输出 JSON，不要解释

输出格式：
{
  "entities": [{"name": "平安银行", "type": "公司"}],
  "relations": [{"head": "平安银行", "relation": "披露", "tail": "净利润", "value": "281.95亿元"}]
}

文本：
{text}
"""


def extract(text):
    prompt = PROMPT.replace("{text}", text[:1500])
    try:
        resp = llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = resp.choices[0].message.content
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            return None
        return json.loads(m.group())
    except Exception:
        return None


if __name__ == "__main__":
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]
    print("片段数:", len(chunks))

    G = nx.DiGraph()
    entity_map = {}  # name -> set of chunk ids

    for i, c in enumerate(chunks):
        if i >= 800:
            break
        result = extract(c["text"])
        if not result:
            continue
        for e in result.get("entities", []):
            name = e.get("name", "").strip()
            if not name:
                continue
            G.add_node(name, type=e.get("type", "未知"))
            entity_map.setdefault(name, set()).add(i)
        for r in result.get("relations", []):
            h = r.get("head", "").strip()
            t = r.get("tail", "").strip()
            if h and t:
                G.add_edge(h, t, relation=r.get("relation", ""), value=r.get("value", ""))
        if (i + 1) % 100 == 0:
            print(f"已处理 {i+1}/{len(chunks)}, 实体 {G.number_of_nodes()}, 关系 {G.number_of_edges()}")

    print("实体:", G.number_of_nodes(), "关系:", G.number_of_edges())

    os.makedirs(os.path.dirname(GRAPH_FILE), exist_ok=True)
    data = {
        "nodes": [{"name": n, **G.nodes[n]} for n in G.nodes],
        "edges": [{"head": u, "tail": v, **G.edges[u, v]} for u, v in G.edges],
        "entity_map": {k: list(v) for k, v in entity_map.items()}
    }
    with open(GRAPH_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("已保存:", GRAPH_FILE)