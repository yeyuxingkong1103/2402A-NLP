# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Graph RAG 优化任务
模块：RAG 评估（优化版 - 字符级 + 数字匹配）
功能：改进版 context_precision / context_recall
"""

import re
from typing import List, Set


class CustomRAGEvaluator:
    """自定义 RAG 评估器（优化版）"""

    def _tokenize(self, text: str) -> Set[str]:
        """字符级 n-gram 分词（对数字友好）"""
        if not text:
            return set()
        text = str(text)
        tokens = set()
        # 1. 中文 2-gram
        chinese = re.findall(r"[\u4e00-\u9fa5]+", text)
        for seg in chinese:
            for i in range(len(seg) - 1):
                tokens.add(seg[i:i+2])
        # 2. 数字 / 百分号
        numbers = re.findall(r"\d+(?:\.\d+)?%?", text)
        for n in numbers:
            tokens.add(n)
        # 3. 英文
        english = re.findall(r"[A-Za-z]{2,}", text)
        for e in english:
            tokens.add(e.lower())
        return tokens

    def context_precision(self, retrieved_contexts: List[str],
                          ground_truth: str) -> float:
        if not retrieved_contexts:
            return 0.0
        gt_tokens = self._tokenize(ground_truth)
        if not gt_tokens:
            return 0.0
        relevant = 0
        for ctx in retrieved_contexts:
            ctx_tokens = self._tokenize(ctx)
            if not ctx_tokens:
                continue
            overlap = len(ctx_tokens & gt_tokens) / max(len(gt_tokens), 1)
            if overlap >= 0.03:
                relevant += 1
        return relevant / len(retrieved_contexts)

    def context_recall(self, retrieved_contexts: List[str],
                       ground_truth: str) -> float:
        if not retrieved_contexts:
            return 0.0
        gt_tokens = self._tokenize(ground_truth)
        if not gt_tokens:
            return 0.0
        all_ctx_tokens = set()
        for ctx in retrieved_contexts:
            all_ctx_tokens |= self._tokenize(ctx)
        return len(gt_tokens & all_ctx_tokens) / len(gt_tokens)


if __name__ == "__main__":
    ev = CustomRAGEvaluator()
    contexts = ["平安银行2019年实现营业收入1,379亿元", "其他信息..."]
    gt = "1,379亿元"
    print("context_precision:", ev.context_precision(contexts, gt))
    print("context_recall:", ev.context_recall(contexts, gt))
