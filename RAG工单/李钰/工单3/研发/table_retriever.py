# -*- coding: utf-8 -*-
"""
表格专用检索器
工单编号: 人工智能 NLP-RAG-PDF 文档的表格解析及检索优化

核心逻辑:
    1. 判定问题是否需要表格检索 (含发行/募集/关联方/持股等关键词)
    2. 表格检索: 列名匹配 + 内容匹配
    3. 返回完整表格 + 命中行 + 格式化 summary
"""
import os
import json
import logging
import numpy as np
from typing import List, Dict, Tuple

import config_v3 as config

logger = logging.getLogger(__name__)

# 表格类问题关键词 (命中则优先走表格检索)
TABLE_QUERY_KEYWORDS = [
    # 发行相关
    "发行股数", "发行价格", "总股本", "比例", "面值", "发行后",
    "股本", "发行概况", "发行数量",
    # 募集资金
    "募集资金", "募集", "投资项目", "资金用途", "资金运用", "用途",
    # 关联方
    "关联方", "关联", "控制关系", "不存在控制", "控制",
    # 持股/股东
    "持股", "股东", "股权", "比例", "关系",
    # 财务数据 (也常出现在表格)
    "收入分别", "比重", "占主营业务",
]


def is_table_query(query: str) -> bool:
    """判定问题是否为表格密集型"""
    hits = [kw for kw in TABLE_QUERY_KEYWORDS if kw in query]
    if hits:
        logger.info(f"[表格路由] 问题含关键词: {hits}")
        return True
    return False


class TableRetriever:
    """表格专用检索器"""

    def __init__(self):
        self.tables: List[Dict] = []

    def load_tables(self, tables: List[Dict]):
        self.tables = tables
        logger.info(f"[表格检索] 加载 {len(tables)} 张表格")

    def _score_table(self, query_tokens: List[str], table: Dict) -> Tuple[float, Dict]:
        """
        对单张表格打分

        Returns:
            (总分, 命中详情)
        """
        columns = [c for c in table.get("columns", []) if c]
        rows = table.get("rows", [])
        summary = table.get("summary", "")
        text_blocks = table.get("text_blocks", [])

        # 1. 列名匹配 (权重 0.4)
        col_hits = 0
        for col in columns:
            for tok in query_tokens:
                if tok in col:
                    col_hits += 1
        col_score = min(col_hits / max(len(query_tokens), 1), 1.0)

        # 2. 内容匹配 (权重 0.4)
        content_score = 0.0
        for tb in text_blocks:
            hits = sum(1 for tok in query_tokens if tok in tb)
            if hits:
                content_score = max(content_score, hits / max(len(query_tokens), 1))
        # 也检查 summary
        summary_hits = sum(1 for tok in query_tokens if tok in summary)
        summary_score = summary_hits / max(len(query_tokens), 1)
        content_score = max(content_score, summary_score)

        # 3. 公司匹配 (权重 0.2)
        company_bonus = 0.0
        # 公司匹配已在路由层处理, 这里作为加分

        final = 0.4 * col_score + 0.4 * content_score + 0.2 * company_bonus
        return final, {
            "col_score": col_score,
            "content_score": content_score,
            "matched_cols": [c for c in columns
                            for t in query_tokens if t in c],
        }

    def search(self, query: str, company_filter: str = None,
               top_k: int = 3) -> List[Dict]:
        """
        表格检索

        Args:
            query: 用户问题
            company_filter: 限定搜索的公司名 (来自路由)
            top_k: 返回表格数量

        Returns:
            List[{"table": Dict, "score": float, "hit_rows": List[Dict]}]
        """
        try:
            import jieba
            tokens = [t for t in jieba.cut(query) if len(t.strip()) > 0]
        except ImportError:
            tokens = list(query)

        candidates = self.tables
        if company_filter:
            candidates = [t for t in candidates
                          if t.get("company", "") == company_filter]

        scored = []
        for table in candidates:
            score, details = self._score_table(tokens, table)
            if score > 0:
                # 找出命中的行
                hit_rows = []
                for r_idx, row in enumerate(table.get("rows", [])):
                    row_text = " ".join(row)
                    hit_tokens = [t for t in tokens if t in row_text]
                    if hit_tokens:
                        hit_rows.append({
                            "row_index": r_idx,
                            "row": row,
                            "hit_tokens": hit_tokens,
                        })
                scored.append({
                    "table": table,
                    "score": round(score, 4),
                    "hit_rows": hit_rows,
                    "matched_cols": details["matched_cols"],
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def format_for_llm(self, results: List[Dict]) -> str:
        """将检索结果格式化为 LLM 友好的表格文本"""
        parts = []
        for i, r in enumerate(results):
            t = r["table"]
            header = " | ".join([c for c in t["columns"] if c])
            rows_text = []
            for row in t["rows"]:
                rows_text.append(" | ".join(row))

            part = f"【表格 {i+1}: {t['table_name']} (第{t['page']}页, {t['company']})】\n"
            part += f"表头: {header}\n"
            part += "数据:\n" + "\n".join(rows_text)

            # 如果有命中行, 高亮
            hit_row_indices = [hr["row_index"] for hr in r["hit_rows"]]
            if hit_row_indices:
                part += f"\n命中行: {hit_row_indices} (关键词: {r['matched_cols']})"

            parts.append(part)

        return "\n\n".join(parts) if parts else ""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 加载预设表格
    with open(config.PRESET_TABLES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    retriever = TableRetriever()
    retriever.load_tables(data["tables"])

    tests = [
        "武汉力源信息技术股份有限公司本次发行股数是多少, 占发行后总股本的比例是多少?",
        "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目?",
        "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁, 持股比例和本公司关系是什么?",
    ]
    for q in tests:
        print(f"\n问题: {q}")
        print(f"  表格查询: {is_table_query(q)}")
        results = retriever.search(q, top_k=2)
        for r in results:
            print(f"  → [{r['table']['table_name']}] score={r['score']}, "
                  f"命中行={[hr['row_index'] for hr in r['hit_rows']]}")
        print(retriever.format_for_llm(results))
