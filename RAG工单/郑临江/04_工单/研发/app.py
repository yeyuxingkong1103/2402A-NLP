# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
主程序：解析 PDF（文本 + 表格 + 图像），图像用多模态模型解析语义，
并支持图像类问题（id 5、6）的检索回复。
用法：python app.py
"""
import config
from pdf_parser import extract_pdf_content, extract_images, chunk_text
from image_parser import build_image_descriptions
from retriever import BM25Retriever
from llm import LLM, build_rag_prompt


class ImageRagQA:
    def __init__(self):
        self.llm = LLM()
        self.retriever = BM25Retriever()

    def build(self, pdf_paths):
        all_docs = []
        page_text_map = {}
        for pdf in pdf_paths:
            print(f"[RAG] 解析 PDF：{pdf}")
            text_blocks, table_blocks = extract_pdf_content(pdf)
            for t in text_blocks:
                all_docs.extend(chunk_text(t, config.CHUNK_SIZE, config.CHUNK_OVERLAP))
                # 记录每页文字，供图像上下文降级
                head = t.split("\n", 1)[0]
                try:
                    pno = int(head.replace("【第", "").replace("页】", ""))
                    page_text_map[pno] = t
                except Exception:
                    pass
            all_docs.extend(table_blocks)

            # 图像提取与语义解析
            infos = extract_images(pdf, config.IMAGE_DIR)
            print(f"    提取图片 {len(infos)} 张")
            descs = build_image_descriptions(infos, page_text_map, config.CLIP_LABELS)
            for desc, path, pno in descs:
                all_docs.append(f"【第{pno}页·图像】\n{desc}")
        self.retriever.build_index(all_docs)
        print(f"[RAG] 索引构建完成，共 {len(all_docs)} 个检索块。")

    def ask(self, question):
        hits = self.retriever.search(question, top_k=config.TOP_K)
        answer = self.llm.generate(build_rag_prompt(question, hits))
        return {"question": question, "answer": answer, "hits": hits}


def run_test(qa):
    print("\n================ 图像内容解析检索演示 ================")
    for q in config.QUESTIONS:
        res = qa.ask(q["question"])
        print(f"\n--- 问题(id={q['id']}) ---")
        print("Q:", q["question"])
        print("A:", res["answer"][:220].replace("\n", " "))
    print("\n================ 演示结束 ================")


if __name__ == "__main__":
    qa = ImageRagQA()
    qa.build([config.PDF1, config.PDF2])
    run_test(qa)
