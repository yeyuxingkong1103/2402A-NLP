# -*- coding: utf-8 -*-
"""
多 PDF 文档管理器
工单编号: 人工智能 NLP-RAG-PDF 文档的表格解析及检索优化

功能:
    1. 统一管理多个 PDF 文档的加载
    2. 根据公司别名路由到对应文档
    3. 每个 PDF 独立解析, 合并结果
    4. 缺失 PDF 降级处理 (预设数据)
"""
import os
import json
import logging
from typing import Dict, List, Optional

import config_v3 as config

logger = logging.getLogger(__name__)


class PDFDoc:
    """单个 PDF 文档的包装"""
    def __init__(self, path: str, company: str, aliases: List[str]):
        self.path = path
        self.company = company
        self.aliases = aliases
        self.exists = os.path.exists(path)
        self.chunks: List[Dict] = []          # 文本块
        self.tables: List[Dict] = []          # 结构化表格
        self.text_index = None                 # 文本检索索引
        self.table_index = None                # 表格检索索引

    def __repr__(self):
        return f"PDFDoc({self.company}, path={self.path}, exists={self.exists})"


class MultiPDFManager:
    """多文档统一管理器"""

    def __init__(self):
        self.docs: List[PDFDoc] = []
        self._init_docs()

    def _init_docs(self):
        for spec in config.PDF_DOCS:
            doc = PDFDoc(spec["path"], spec["company"], spec["aliases"])
            self.docs.append(doc)
            status = "已找到" if doc.exists else "缺失, 将降级处理"
            logger.info(f"[多文档] {doc.company}: {status} ({doc.path})")

    def get_doc_by_company(self, name: str) -> Optional[PDFDoc]:
        """根据公司名/别名查找文档"""
        name_lower = name.lower()
        for doc in self.docs:
            if doc.company.lower() == name_lower:
                return doc
            for alias in doc.aliases:
                if alias.lower() == name_lower or name_lower in alias.lower():
                    return doc
        return None

    def route_query(self, query: str) -> List[PDFDoc]:
        """
        根据问题路由到相关文档

        Returns:
            List[PDFDoc] - 需要检索的文档列表
        """
        routed = []
        for doc in self.docs:
            # 问题中包含公司名或任一别名 → 必选
            if doc.company in query:
                routed.append(doc)
                continue
            matched = any(alias in query for alias in doc.aliases)
            if matched:
                routed.append(doc)

        if routed:
            logger.info(f"[路由] 问题 '{query[:30]}...' → {[d.company for d in routed]}")
            return routed

        # 无明确匹配 → 返回所有存在的文档
        return [d for d in self.docs if d.exists or self._has_preset(d)]

    def _has_preset(self, doc: PDFDoc) -> bool:
        return doc.company == "武汉力源信息技术股份有限公司" and \
               os.path.exists(config.PRESET_TABLES_PATH)

    def load_preset_tables(self, doc: PDFDoc) -> List[Dict]:
        """加载预设表格数据"""
        if not os.path.exists(config.PRESET_TABLES_PATH):
            return []
        try:
            with open(config.PRESET_TABLES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            tables = data.get("tables", [])
            logger.info(f"[预设] 加载 {len(tables)} 张表格 → {doc.company}")
            return tables
        except Exception as e:
            logger.error(f"加载预设表格失败: {e}")
            return []


def get_manager() -> MultiPDFManager:
    """获取单例"""
    if not hasattr(get_manager, "_instance"):
        get_manager._instance = MultiPDFManager()
    return get_manager._instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mgr = get_manager()
    for doc in mgr.docs:
        print(doc)

    # 路由测试
    tests = [
        "武汉力源信息技术股份有限公司本次发行股数是多少?",
        "武汉兴图新科电子股份有限公司注册资本是多少?",
        "关联方持股比例",  # 模糊 → 所有文档
    ]
    for q in tests:
        docs = mgr.route_query(q)
        print(f"\n问题: {q}")
        print(f"  → {[d.company for d in docs]}")
