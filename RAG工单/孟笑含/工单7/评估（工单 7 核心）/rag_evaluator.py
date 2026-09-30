# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块：RAG评估模块
功能：评估RAG系统检索与生成质量
"""

from typing import List, Dict, Any
import re


class RAGEvaluator:
    """RAG评估器"""
    
    def __init__(self):
        pass
    
    def evaluate_answer_relevance(self, question: str, answer: str, 
                                   contexts: List[Dict]) -> Dict[str, float]:
        """
        评估答案相关性
        :return: 包含各项指标的字典
        """
        # 1. 关键词覆盖率
        keywords = self._extract_keywords(question)
        answer_lower = answer.lower()
        covered = sum(1 for kw in keywords if kw.lower() in answer_lower)
        keyword_coverage = covered / len(keywords) if keywords else 0.0
        
        # 2. 答案长度合理性
        length_score = min(1.0, len(answer) / 50) if answer else 0.0
        
        # 3. 是否包含数值信息（针对数值型问题）
        has_numbers = bool(re.search(r'\d+', answer))
        number_score = 1.0 if has_numbers else 0.0
        
        return {
            "keyword_coverage": keyword_coverage,
            "length_score": length_score,
            "number_score": number_score,
            "overall": (keyword_coverage * 0.5 + length_score * 0.25 + number_score * 0.25)
        }
    
    def evaluate_retrieval(self, question: str, 
                          retrieved_contexts: List[Dict]) -> Dict[str, float]:
        """
        评估检索质量
        """
        # 1. 检索数量
        num_results = len(retrieved_contexts)
        
        # 2. 最高相似度得分
        max_score = max([ctx.get("score", 0) for ctx in retrieved_contexts]) if retrieved_contexts else 0
        
        # 3. 平均相似度得分
        avg_score = sum([ctx.get("score", 0) for ctx in retrieved_contexts]) / num_results if num_results else 0
        
        # 4. 关键词命中率
        keywords = self._extract_keywords(question)
        hit_count = 0
        for ctx in retrieved_contexts:
            content = ctx.get("content", "")
            for kw in keywords:
                if kw in content:
                    hit_count += 1
                    break
        hit_rate = hit_count / num_results if num_results else 0
        
        return {
            "num_results": num_results,
            "max_similarity": max_score,
            "avg_similarity": avg_score,
            "keyword_hit_rate": hit_rate,
            "overall": (max_score * 0.4 + hit_rate * 0.6)
        }
    
    def compare_with_llm_only(self, question: str, 
                              rag_answer: str, 
                              llm_answer: str) -> Dict[str, Any]:
        """
        对比RAG答案与纯LLM答案
        """
        rag_eval = self.evaluate_answer_relevance(question, rag_answer, [])
        llm_eval = self.evaluate_answer_relevance(question, llm_answer, [])
        
        return {
            "question": question,
            "rag_answer": rag_answer,
            "llm_answer": llm_answer,
            "rag_score": rag_eval,
            "llm_score": llm_eval,
            "winner": "RAG" if rag_eval["overall"] > llm_eval["overall"] else "LLM",
            "improvement": rag_eval["overall"] - llm_eval["overall"]
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        stop_words = {"的", "了", "是", "在", "有", "和", "与", "及", "根据", 
                      "报告期内", "分别", "多少", "哪些", "哪个", "什么", "怎么"}
        words = re.findall(r'[\u4e00-\u9fa5]{2,}|\d+', text)
        return [w for w in words if w not in stop_words]