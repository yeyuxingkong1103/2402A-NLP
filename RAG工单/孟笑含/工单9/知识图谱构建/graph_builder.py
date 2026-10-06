# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
模块：知识图谱构建
功能：从 PDF 抽取实体/关系，构建 NetworkX 图
"""

import os
import re
import json
import fitz
import torch
import networkx as nx
from typing import List, Dict, Any
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

os.environ['HF_HUB_DISABLE_XET'] = '1'


class GraphBuilder:
    """知识图谱构建器（Qwen2.5-VL 抽取 + NetworkX）"""

    ENTITY_TYPES = ["公司", "人物", "机构", "指标", "数值", "年份", "地点", "产品"]
    RELATION_TYPES = ["属于", "任职", "持股", "控股", "报告", "实现", "同比", "是", "拥有"]

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        print(f"正在加载 LLM：{model_name}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.graph = nx.MultiDiGraph()
        print("✅ LLM 加载完成")

    def extract_from_text(self, text: str, doc: str, page: int) -> List[Dict]:
        """用 LLM 从文本里抽取三元组"""
        prompt = f"""从下面文本中抽取知识图谱三元组（头实体，关系，尾实体）。

要求：
1. 只抽取明确的实体和关系
2. 实体类型：{', '.join(self.ENTITY_TYPES)}
3. 关系类型：{', '.join(self.RELATION_TYPES)}
4. 输出 JSON 数组，每个元素：{{"head": "实体1", "head_type": "类型", "relation": "关系", "tail": "实体2", "tail_type": "类型"}}
5. 最多抽取 20 个三元组

文本：
{text[:3000]}

只输出 JSON 数组，不要解释："""

        messages = [{"role": "user", "content": prompt}]
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=[text_input], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=2048)

        output = self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

        # 解析 JSON
        try:
            match = re.search(r"\[.*\]", output, re.DOTALL)
            if match:
                triples = json.loads(match.group(0))
                for t in triples:
                    t["doc"] = doc
                    t["page"] = page
                return triples
        except Exception as e:
            print(f"  解析失败：{e}")
        return []

    def build_from_pdfs(self, pdf_paths: List[str], max_pages_per_doc: int = 30):
        """从多个 PDF 构建图谱"""
        total_triples = 0
        for pdf_path in pdf_paths:
            doc_name = os.path.basename(pdf_path)
            print(f"\n处理：{doc_name}")
            doc = fitz.open(pdf_path)
            n_pages = min(len(doc), max_pages_per_doc)

            for page_num in range(n_pages):
                page = doc[page_num]
                text = page.get_text()[:3000]
                if len(text) < 100:
                    continue

                try:
                    triples = self.extract_from_text(text, doc_name, page_num + 1)
                except Exception as e:
                    print(f"    抽取异常：{e}")
                    triples = []
                for t in triples:
                    # 校验字段完整性
                    if not all(k in t for k in ["head", "tail", "relation"]):
                        continue
                    if not t["head"] or not t["tail"] or not t["relation"]:
                        continue
                    self.graph.add_node(t["head"], type=t.get("head_type", "未知"))
                    self.graph.add_node(t["tail"], type=t.get("tail_type", "未知"))
                    self.graph.add_edge(t["head"], t["tail"],
                                        relation=t["relation"], doc=doc_name, page=page_num + 1)
                total_triples += len(triples)
                print(f"  第 {page_num+1}/{n_pages} 页：抽取 {len(triples)} 三元组")

            doc.close()

        print(f"\n✅ 图谱构建完成")
        print(f"  节点数：{self.graph.number_of_nodes()}")
        print(f"  边数：{self.graph.number_of_edges()}")
        print(f"  三元组总数：{total_triples}")

    def save_graph(self, path: str = "graph.json"):
        """保存图谱为 JSON"""
        data = nx.node_link_data(self.graph)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ 图谱已保存到 {path}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
        }


if __name__ == "__main__":
    # 只抽取 2 个招股书的前 10 页（快速测试）
    builder = GraphBuilder()
    builder.build_from_pdfs([
        "./data/招股说明书1.pdf",
        "./data/招股说明书2.pdf",
    ], max_pages_per_doc=10)
    builder.save_graph("graph_test.json")
    print(builder.get_stats())
