# main_lightrag.py 工单12 LightRAG主程序
import os
import json
from lightrag import LightRAG, QueryParam
from config_lightrag import *

def init_lightrag():
    """初始化LightRAG实例，本地BGE向量模型 + Ollama Qwen2:7b"""
    rag = LightRAG(
        working_dir=CACHE_DIR,
        embedding_model=EMBEDDING_MODEL_PATH,
        llm_model=LLM_MODEL,
        llm_model_host=LLM_HOST,
        entity_extract_prompt=ENTITY_EXTRACT_PROMPT,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return rag

def build_knowledge_base(rag):
    """加载pdf文档，构建LightRAG知识库（图谱+向量）"""
    pdf_files = [os.path.join(PDF_DIR,f) for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    for pdf_path in pdf_files:
        print(f"开始加载文档：{pdf_path}")
        with open(pdf_path,"rb") as f:
            raw_bytes = f.read()
        rag.insert(raw_bytes)
    print("✅知识库构建完成，支持增量文档插入，无需全量重建")

def query_normal_rag(rag, question:str):
    """普通RAG naive模式，仅向量检索，无图谱"""
    param = QueryParam(mode="naive")
    resp = rag.query(question, param=param)
    return resp

def query_lightrag_hybrid(rag, question:str):
    """LightRAG混合检索：局部语义+全局知识图谱推理"""
    param = QueryParam(mode="hybrid")
    resp = rag.query(question, param=param)
    return resp

def batch_run_test(rag, question_json_path, output_save_path):
    """批量跑全部测试问题，保存两套RAG的回答结果"""
    with open(question_json_path,"r",encoding="utf-8") as f:
        q_list = json.load(f)
    result_list = []
    for item in q_list:
        qid = item["id"]
        q = item["question"]
        print(f"\n====处理问题 id:{qid} ====")
        ans_normal = query_normal_rag(rag, q)
        ans_light = query_lightrag_hybrid(rag, q)
        result_list.append({
            "id": qid,
            "question": q,
            "ans_normal_rag": ans_normal,
            "ans_lightrag": ans_light
        })
    with open(output_save_path,"w",encoding="utf-8") as f:
        json.dump(result_list,f,ensure_ascii=False,indent=2)
    print(f"\n✅批量问答结果保存至：{output_save_path}")

if __name__ == "__main__":
    rag_engine = init_lightrag()
    # 构建知识库，首次运行打开；后续增量测试可注释
    build_knowledge_base(rag_engine)
    # 批量执行9条测试问题
    batch_run_test(rag_engine,"./test_questions.json","./batch_query_result.json")
