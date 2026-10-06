# ============================================================
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
# 项目名称：Graph RAG 优化
# 文件：graph_rag_engine.py
# 说明：一步到位版优化
#       向量召回 k=30 → Reranker top15；图谱三元组 max=50 +
#       实体过滤；实体识别 jieba + 连续片段双路匹配
# ============================================================
import json
import networkx as nx
import jieba
from rag_engine import RAGEngine


class GraphRAGEngine(RAGEngine):
    """向量检索 + 图谱三元组，双路融合"""

    def __init__(self, model_path, db_path, embed_path, reranker_path, kg_path):
        super().__init__(model_path, db_path, embed_path, reranker_path)

        print(f"加载知识图谱：{kg_path}")
        with open(kg_path, "r", encoding="utf-8") as f:
            kg_data = json.load(f)

        self.kg = nx.DiGraph()
        for edge in kg_data["edges"]:
            self.kg.add_edge(
                edge["source"], edge["target"],
                relation=edge.get("relation", ""),
            )
        self.node_set = set(self.kg.nodes())
        print(f"图谱加载完成：{self.kg.number_of_nodes()} 节点 / {self.kg.number_of_edges()} 边")

    # ---------- 实体识别：连续片段 + jieba 双路 ----------
    def _extract_entities(self, question):
        entities = []
        # 1. 连续片段匹配（长度 6→2）
        for n in range(6, 1, -1):
            for i in range(len(question) - n + 1):
                sub = question[i:i+n]
                if sub in self.node_set and sub not in entities:
                    entities.append(sub)
        # 2. jieba 分词匹配
        for tok in jieba.cut(question):
            if len(tok) >= 2 and tok in self.node_set and tok not in entities:
                entities.append(tok)
        return entities

    # ---------- 图谱三元组抽取 + 实体过滤 ----------
    def _get_graph_context(self, entities, question, max_triples=50):
        """抽取三元组，只保留与问题实体相关、且与问题关键词有交集的"""
        raw = []
        seen = set()
        for ent in entities:
            for _, v, d in self.kg.out_edges(ent, data=True):
                key = (ent, d.get("relation", ""), v)
                if key not in seen:
                    seen.add(key)
                    raw.append(f"{ent} | {d.get('relation','')} | {v}")
            for u, _, d in self.kg.in_edges(ent, data=True):
                key = (u, d.get("relation", ""), ent)
                if key not in seen:
                    seen.add(key)
                    raw.append(f"{u} | {d.get('relation','')} | {ent}")

        # 过滤：三元组必须至少包含一个识别出的实体
        if entities:
            filtered = [t for t in raw if any(e in t for e in entities)]
        else:
            filtered = raw

        # 优先保留实体数多的三元组
        def score(t):
            return sum(1 for e in entities if e in t)

        filtered.sort(key=score, reverse=True)
        return filtered[:max_triples]

    # ---------- 重写 ask ----------
    def ask(self, question, history=None,
            retrieval_mode="hybrid", hybrid_weight=0.5,
            rerank_method="bge"):
        """返回 (answer, docs, graph_triples)"""
        is_valid, cleaned, msg = self._validate_input(question)
        if not is_valid:
            return f"【输入错误】{msg}", [], []

        if history and self._needs_resolution(cleaned):
            try:
                cleaned = self._resolve_query(cleaned, history)
            except Exception:
                pass

        lang = self._detect_language(cleaned)
        if lang != "zh":
            cleaned = self._prepare_question(cleaned)

        # 实体识别 + 图谱三元组
        entities = self._extract_entities(cleaned)
        graph_triples = self._get_graph_context(entities, cleaned, max_triples=50) if entities else []

        # 向量检索：k=30 扩大召回
        search_query = cleaned if len(cleaned) <= 40 else self._rewrite_query(cleaned)
        candidates = self._retrieve(
            search_query, mode=retrieval_mode, weight=hybrid_weight, k=30,
        )

        seen = set()
        unique = []
        for d in candidates:
            key = d.page_content[:80]
            if key not in seen:
                seen.add(key)
                unique.append(d)
        candidates = unique

        # 图像描述专门检索
        image_kw = ["图", "图表", "结构图", "柱状", "饼图", "增长图"]
        image_docs = []
        if any(kw in cleaned for kw in image_kw):
            try:
                image_docs = self.vectordb.similarity_search(
                    search_query, k=2, filter={"source": "图像描述"},
                )
            except Exception:
                image_docs = []

        # Reranker 精排：top_k=15
        top_docs = self._rerank(search_query, candidates, top_k=15, method=rerank_method)
        top_docs = image_docs + top_docs

        vector_context = "\n\n".join([d.page_content for d in top_docs])

        graph_context = ""
        if graph_triples:
            graph_context = "\n\n【知识图谱三元组（主体 | 关系 | 客体）】\n" + "\n".join(graph_triples)

        full_context = vector_context + graph_context

        lang_instr = "请用中文回答。" if lang == "zh" else "请用英文回答。"

        prompt = f"""你是金融年报问答助手。请严格基于以下上下文回答问题。

上下文包含两部分：
1. 文本片段（来自年报原文）
2. 知识图谱三元组（格式：主体 | 关系 | 客体）

重要规则：
1. 文本片段优先，图谱三元组作为实体关系的补充。
2. 回答中的数字必须严格来自上下文，不得自行推算。
3. 如果上下文中找不到信息，明确回答“上下文未提供相关信息”。
4. {lang_instr}

上下文：
{full_context}

问题：{question}

回答："""

        answer = self._generate(prompt, max_new_tokens=512)

        if lang != "zh":
            answer = self._fix_company_names_in_answer(answer)

        return answer, top_docs, graph_triples