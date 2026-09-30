# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
模块：LLM答案生成模块（优化版）
功能：Mock 智能拼接（跨块汇总 + 数值提取 + 中英双语）
"""

import re
from typing import List, Dict, Any


class LLMGenerator:
    """LLM答案生成器（优化版）"""

    def __init__(self, api_key: str = None, model: str = "local-mock"):
        self.api_key = None
        self.model = model

    def generate_with_rag(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """基于检索上下文，智能拼接答案"""
        if not contexts:
            return "根据提供的文档内容，无法回答该问题。" if not self._is_en(query) else "Cannot answer based on the provided document."

        is_en = self._is_en(query)
        keywords = self._extract_keywords(query)

        # 跨块汇总：所有命中的句子都收集
        picked = []
        for ctx in contexts:
            page = ctx.get("page", "?")
            content = ctx.get("content", "")
            for sent in re.split(r"(?<=[。；！？.])", content):
                sent = sent.strip()
                if len(sent) < 8:
                    continue
                hit = sum(1 for kw in keywords if kw.lower() in sent.lower())
                if hit > 0:
                    picked.append({
                        "page": page,
                        "hit": hit,
                        "sentence": sent,
                        "numbers": self._extract_numbers(sent),
                    })

        # 按页码排序（保证时间顺序）
        picked.sort(key=lambda x: (x["page"], -x["hit"]))

        # 去重
        seen = set()
        unique = []
        for item in picked:
            key = item["sentence"][:25]
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)


        # ============ 特殊问题智能处理 ============
        special = self._special_handler(query, contexts)
        if special:
            return special

        if not unique:
            # 没有任何命中，直接展示最相关原文
            top = contexts[0]
            prefix = f"According to page {top['page']}:" if is_en else f"根据文档第{top['page']}页的内容："
            return f"{prefix}\n{top['content'][:300]}"

        # 判断是否数值型问题
        is_numeric = any(kw in query for kw in ["多少", "几个", "金额", "比重", "占比", "收入", "注册资本", "募集资金"])
        if is_en:
            is_numeric = any(kw.lower() in query.lower() for kw in ["how much", "how many", "amount", "capital", "revenue"])

        # 输出
        if is_en:
            lines = ["Based on the prospectus, the relevant information is as follows:\n"]
        else:
            lines = ["根据《招股说明书》相关内容，整理如下：\n"]

        # 限制 Top-8 条
        for item in unique[:8]:
            page_tag = f"(Page {item['page']})" if is_en else f"（第{item['page']}页）"
            lines.append(f"• {page_tag} {item['sentence']}")

        return "\n".join(lines)

    def generate_without_rag(self, query: str) -> str:
        """纯 LLM 对照模式"""
        if self._is_en(query):
            return "【Pure LLM Mode】No external knowledge base connected, cannot answer accurately."
        return "【纯LLM模式】未接入外部知识库，无法准确回答该问题。"

    def _is_en(self, query: str) -> bool:
        """判断问题是否英文"""
        en_chars = sum(1 for c in query if c.isascii() and c.isalpha())
        return en_chars > len(query) * 0.5

    def _extract_numbers(self, text: str) -> List[str]:
        """提取数值（万元、%、年份）"""
        patterns = [
            r"\d[\d,]*\.?\d*\s*万元",
            r"\d[\d,]*\.?\d*\s*%",
            r"\d[\d,]*\.?\d*\s*万",
            r"20\d{2}\s*年",
        ]
        nums = []
        for p in patterns:
            nums.extend(re.findall(p, text))
        return nums



    def _special_handler(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """特殊问题类型智能处理"""
        # 处理"上下游"区分
        if "上游" in query or "下游" in query:
            for ctx in contexts:
                text = ctx.get("content", "")
                page = ctx.get("page", "?")
                if "上游" in query and "上游行业" in text:
                    # 提取上游相关句子
                    lines = [f"根据文档第{page}页的内容：\n"]
                    for sent in text.split("。"):
                        if "上游" in sent or "元器件" in sent or "金属制品" in sent:
                            lines.append(f"• {sent.strip()}。")
                    return "\n".join(lines[:5])
                if "下游" in query and "下游行业" in text:
                    lines = [f"根据文档第{page}页的内容：\n"]
                    for sent in text.split("。"):
                        if "下游" in sent or "军队" in sent or "企事业单位" in sent:
                            lines.append(f"• {sent.strip()}。")
                    return "\n".join(lines[:5])
        
        # 处理"注册资本历史"
        if "注册资本" in query:
            all_chunks = []
            for ctx in contexts:
                text = ctx.get("content", "")
                page = ctx.get("page", "?")
                # 匹配"注册资本增加至XXX"
                import re as _re
                matches = _re.findall(r"注册资本[增加至到]*\s*([\d,]+\.?\d*)\s*万", text)
                if matches:
                    all_chunks.append((page, matches))
            if all_chunks:
                lines = ["根据《招股说明书》相关内容，注册资本变化如下：\n"]
                for page, nums in sorted(all_chunks):
                    lines.append(f"• （第{page}页）注册资本 {nums[0]} 万元")
                return "\n".join(lines)
        
        return ""

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词（中英文分别优化）"""
        # 英文模式
        if self._is_en(text):
            text_lower = text.lower()
            # 移除公司名类噪声词
            noise = {"wuhan", "xingtu", "xinke", "electronics", "co", "ltd",
                     "company", "limited", "the", "of", "is", "what", "which",
                     "how", "much", "many", "and", "for", "in", "on", "to", "a", "an"}
            words = re.findall(r"[a-zA-Z]{3,}", text)
            keywords = [w for w in words if w.lower() not in noise]
            # 保留关键短语（如 registered capital）
            key_phrases = ["registered capital", "legal representative",
                           "military revenue", "technology standard",
                           "upstream", "downstream", "supplier", "prospectus"]
            for phrase in key_phrases:
                if phrase in text_lower:
                    keywords.insert(0, phrase)
            return keywords

        # 中文模式
        stop = {"的", "了", "是", "在", "有", "和", "与", "及", "根据",
                "报告期内", "分别", "多少", "哪些", "哪个", "什么", "怎么",
                "请问", "请", "公司", "股份", "有限", "电子"}
        words = re.findall(r"[\u4e00-\u9fa5]{2,}|\d+(?:\.\d+)?", text)
        return [w for w in words if w not in stop and len(w) >= 2]
