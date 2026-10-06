import torch
import jieba
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


def _jieba_tokenize(text):
    return list(jieba.cut(text))


class RAGEngine:
    def __init__(self, model_path, db_path, embed_path):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_path,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectordb = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings,
        )

        data = self.vectordb.get()
        all_docs = [
            Document(page_content=c, metadata={"id": i})
            for c, i in zip(data["documents"], data["ids"])
        ]
        bm25_retriever = BM25Retriever.from_documents(
            all_docs, preprocess_func=_jieba_tokenize
        )
        bm25_retriever.k = 6

        # 两路检索器分别保存
        self.vector_retriever = self.vectordb.as_retriever(search_kwargs={"k": 6})
        self.bm25_retriever = bm25_retriever

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    def _generate(self, prompt, max_new_tokens=512):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

    def _rewrite_query(self, question):
        prompt = f"""请把下面的问题改写成一句简短的检索查询，保留核心信息（要查什么、哪个主体、哪段时间），去掉“请问”“我想知道”“分别是多少”这类客套词，并把冗长的公司全名简化为“公司”。

问题：{question}

只输出改写后的查询，不要解释，不要加引号。"""
        rewritten = self._generate(prompt, max_new_tokens=64)
        return rewritten.split("\n")[0].strip()

    def ask(self, question):
        # 1. Query 理解
        search_query = self._rewrite_query(question)

        # 2. 两路检索：向量优先，BM25 补充，合计去重后最多 8 块
        vector_docs = self.vector_retriever.invoke(search_query)
        bm25_docs = self.bm25_retriever.invoke(search_query)

        seen = set()
        unique_docs = []
        for d in vector_docs + bm25_docs:
            key = d.page_content[:80]
            if key not in seen:
                seen.add(key)
                unique_docs.append(d)
                if len(unique_docs) >= 8:
                    break

        context = "\n\n".join([d.page_content for d in unique_docs])

        # 3. 生成
        prompt = f"""你是招股说明书问答助手。请严格基于以下上下文回答问题。

重要规则：
1. 注意区分“武汉兴图新科电子股份有限公司”（母公司/发行人）与其下属子公司（如“北京启目”“武汉启目”“华创兴图”等）。
2. 上下文中的数值如果明确标注属于某家子公司，不能当作母公司的数值回答。
3. 关于“军用领域收入占主营业务收入的比重”，这是“直接军方+间接军方”合计占比（如 82.10%、97.31%、94.84%、94.34%），不要用单独的“直接军方占比”或“间接军方占比”代替。
4. 关于“来自军用领域的收入”的四个年度合计值，请严格按原文“报告期内，公司来自军用领域的收入分别为 6,464.51 万元、14,414.16 万元、18,780.67 万元和 4,627.14 万元”这一表述顺序对应 2016 年度、2017 年度、2018 年度、2019 年 1-6 月，不要用其他近似数字（如 18,687.75、4,508.18）替代。
5. 如果上下文中找不到与问题主体匹配的信息，明确回答“上下文未提供相关信息”，不要臆测。

上下文：
{context}

问题：{question}

回答："""
        answer = self._generate(prompt, max_new_tokens=512)
        return answer, unique_docs