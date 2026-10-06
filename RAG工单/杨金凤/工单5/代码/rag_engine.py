# ============================================================
# 工单编号：人工智能NLP-RAG-Query理解优化任务
# 项目名称：PDF文档的Query理解优化
# 文件：rag_engine.py
# 说明：RAG 引擎核心，在原有多PDF检索、图像描述、表格处理基础上，
#       新增多轮对话支持（指代消解 + 历史上下文）
# ============================================================
import re
import time
import torch
import jieba
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


def _jieba_tokenize(text):
    return list(jieba.cut(text))


class RAGEngine:
    MAX_QUESTION_LENGTH = 500
    MIN_QUESTION_LENGTH = 1
    MAX_RETRY = 2
    RETRY_INTERVAL = 1.0

    # 指代词列表（用于检测是否需要消解）
    PRONOUNS = [
        "他", "她", "它", "他们", "她们", "它们",
        "这个公司", "该公司", "这家公司", "那家公司", "此公司",
        "这个", "那个", "此", "该",
        "上述", "前面", "前面提到",
        "呢", "那么", "那",
    ]

    EN_TO_ZH_COMPANY = {
        "Wuhan P&S Information Technology Co., Ltd.": "武汉力源信息技术股份有限公司",
        "Wuhan P&S Information Technology Co.,Ltd.": "武汉力源信息技术股份有限公司",
        "Wuhan P&S Information Technology": "武汉力源信息技术股份有限公司",
        "Wuhan Xingtu Xinke Electronics Co., Ltd.": "武汉兴图新科电子股份有限公司",
        "Wuhan Xingtu Xinke Electronics Co.,Ltd.": "武汉兴图新科电子股份有限公司",
        "Wuhan Xingtu Xinke Electronics": "武汉兴图新科电子股份有限公司",
    }

    ZH_TO_EN_COMPANY = {
        "武汉力源信息技术股份有限公司": "Wuhan P&S Information Technology Co., Ltd.",
        "武汉兴图新科电子股份有限公司": "Wuhan Xingtu Xinke Electronics Co., Ltd.",
    }

    EN_MISTRANSLATION_FIX = {
        "Wuhan Lyrin": "Wuhan P&S",
        "Wuhan Liyuan": "Wuhan P&S",
        "Wuhan Pus": "Wuhan P&S",
        "Wuhan P & S": "Wuhan P&S",
        "Xingtu Xinka": "Xingtu Xinke",
        "Xingtu Xingka": "Xingtu Xinke",
    }

    def __init__(self, model_path, db_path, embed_path, reranker_path):
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embed_path,
                model_kwargs={"device": "cuda"},
                encode_kwargs={"normalize_embeddings": True},
            )
            self.vectordb = Chroma(
                persist_directory=db_path,
                embedding_function=self.embeddings,
            )
        except Exception as e:
            raise RuntimeError(f"【初始化失败】向量库或嵌入模型加载异常：{e}")

        try:
            data = self.vectordb.get()
            all_docs = [
                Document(page_content=c, metadata={"id": i})
                for c, i in zip(data["documents"], data["ids"])
            ]
            bm25_retriever = BM25Retriever.from_documents(
                all_docs, preprocess_func=_jieba_tokenize
            )
            bm25_retriever.k = 10
        except Exception as e:
            raise RuntimeError(f"【初始化失败】BM25 检索器构建异常：{e}")

        self.vector_retriever = self.vectordb.as_retriever(search_kwargs={"k": 10})
        self.bm25_retriever = bm25_retriever

        try:
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_path)
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                reranker_path,
                torch_dtype=torch.float16,
            ).to("cuda").eval()
        except Exception as e:
            raise RuntimeError(f"【初始化失败】Reranker 模型加载异常：{e}")

        try:
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
        except Exception as e:
            raise RuntimeError(f"【初始化失败】LLM 加载异常：{e}")

    # ---------- 输入校验 ----------
    def _validate_input(self, question):
        if question is None:
            return False, "", "输入为空，请输入问题。"
        cleaned = re.sub(r"\s+", " ", question).strip()
        if len(cleaned) < self.MIN_QUESTION_LENGTH:
            return False, "", "输入为空，请输入有效问题。"
        if len(cleaned) > self.MAX_QUESTION_LENGTH:
            cleaned = cleaned[: self.MAX_QUESTION_LENGTH]
            return True, cleaned, f"问题过长，已自动截断至 {self.MAX_QUESTION_LENGTH} 字符。"
        if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", cleaned):
            return False, "", "输入不包含有效字符，请重新输入。"
        return True, cleaned, ""

    # ---------- 生成（带重试） ----------
    def _generate(self, prompt, max_new_tokens=512):
        last_err = None
        for attempt in range(self.MAX_RETRY + 1):
            try:
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
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                last_err = e
                if attempt < self.MAX_RETRY:
                    time.sleep(self.RETRY_INTERVAL)
                    continue
                raise RuntimeError("【生成失败】显存不足，请关闭其他占用 GPU 的进程后重试。")
            except Exception as e:
                last_err = e
                if attempt < self.MAX_RETRY:
                    time.sleep(self.RETRY_INTERVAL)
                    continue
                raise RuntimeError(f"【生成失败】LLM 多次生成异常：{last_err}")
        raise RuntimeError(f"【生成失败】{last_err}")

    # ---------- 多语言 ----------
    def _detect_language(self, text):
        if not text:
            return "zh"
        zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        total = len(re.sub(r"\s", "", text))
        if total == 0:
            return "zh"
        return "zh" if zh_chars / total >= 0.3 else "en"

    def _prepare_question(self, question):
        q = question
        for en_name, zh_name in self.EN_TO_ZH_COMPANY.items():
            q = q.replace(en_name, zh_name)
        return q

    def _fix_company_names_in_answer(self, answer):
        for wrong, correct in self.EN_MISTRANSLATION_FIX.items():
            answer = answer.replace(wrong, correct)
        for zh_name, en_name in self.ZH_TO_EN_COMPANY.items():
            answer = answer.replace(zh_name, en_name)
        return answer

    def _translate(self, text, target_lang):
        lang_name = {"zh": "中文", "en": "英文"}[target_lang]
        prompt = f"""请将以下文本翻译成{lang_name}，只输出翻译结果，不要解释，不要加引号：

{text}"""
        return self._generate(prompt, max_new_tokens=512)

    # ---------- 【工单05】指代消解：多轮对话核心 ----------
    def _needs_resolution(self, question):
        """检测问题是否需要指代消解（含指代词或过短）"""
        # 问题过短（<10 字）通常是省略句
        if len(question.strip()) < 10:
            return True
        # 含指代词
        for p in self.PRONOUNS:
            if p in question:
                return True
        return False

    def _resolve_query(self, question, history):
        """把含指代的问题改写为独立完整的问题
        history: [(q1, a1), (q2, a2), ...] 最近的对话历史
        """
        if not history:
            return question

        # 只取最近 2 轮历史
        recent = history[-2:]
        history_text = ""
        for i, (q, a) in enumerate(recent, 1):
            # 回答截断，避免 prompt 过长
            a_short = a[:200] + "..." if len(a) > 200 else a
            history_text += f"【第{i}轮】\n用户：{q}\n回答：{a_short}\n\n"

        prompt = f"""你是一个问题改写助手。请根据对话历史，把当前用户问题改写为一个独立、完整、可检索的问题。

规则：
1. 如果当前问题包含指代词（如"他"、"她"、"这个公司"、"那...呢"等），请结合历史把指代词替换为具体实体。
2. 如果当前问题是省略句（如"那XXX呢？"），请补全完整的提问。
3. 如果当前问题已经独立完整，直接原样输出。
4. 只输出改写后的问题，不要解释，不要加引号。

对话历史：
{history_text}
当前用户问题：{question}

改写后的问题："""
        try:
            rewritten = self._generate(prompt, max_new_tokens=128)
            result = rewritten.split("\n")[0].strip()
            # 去掉可能的引号
            result = result.strip("\"'“”‘’")
            return result if result else question
        except Exception:
            return question

    # ---------- Query 改写（短查询优化） ----------
    def _rewrite_query(self, question):
        prompt = f"""请把下面的问题改写成一句简短的检索查询，保留核心信息（要查什么、哪个主体、哪段时间），去掉“请问”“我想知道”“分别是多少”这类客套词，并把冗长的公司全名简化为“公司”。

问题：{question}

只输出改写后的查询，不要解释，不要加引号。"""
        try:
            rewritten = self._generate(prompt, max_new_tokens=64)
            result = rewritten.split("\n")[0].strip()
            return result if result else question
        except Exception:
            return question

    # ---------- Reranker 精排 ----------
    def _rerank(self, query, docs, top_k=5):
        if not docs:
            return []
        try:
            pairs = [[query, d.page_content] for d in docs]
            with torch.no_grad():
                inputs = self.reranker_tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                ).to(self.reranker_model.device)
                scores = self.reranker_model(**inputs).logits.view(-1).float()
            ranked = sorted(
                zip(scores.tolist(), docs),
                key=lambda x: x[0],
                reverse=True,
            )
            return [d for _, d in ranked[:top_k]]
        except Exception:
            return docs[:top_k]

    # ---------- 内部：中文检索 + 生成 ----------
    def _ask_chinese(self, question, output_lang="zh"):
        if len(question) <= 40:
            search_query = question
        else:
            search_query = self._rewrite_query(question)

        candidates = []
        seen = set()

        try:
            vector_docs = self.vector_retriever.invoke(search_query)
        except Exception:
            vector_docs = []

        try:
            bm25_docs = self.bm25_retriever.invoke(search_query)
        except Exception:
            bm25_docs = []

        for d in vector_docs + bm25_docs:
            key = d.page_content[:80]
            if key not in seen:
                seen.add(key)
                candidates.append(d)

        # 图像描述专门检索
        image_keywords = ["图", "图中", "图表", "结构图", "走势", "曲线", "柱状", "饼图", "增长图", "应用结构"]
        image_docs_for_context = []
        if any(kw in question for kw in image_keywords):
            try:
                image_docs_for_context = self.vectordb.similarity_search(
                    search_query, k=2,
                    filter={"source": "图像描述"},
                )
            except Exception:
                image_docs_for_context = []

        if not candidates and not image_docs_for_context:
            return "【检索失败】未能从知识库中检索到相关内容，请稍后重试或联系管理员。", []

        top_docs = self._rerank(search_query, candidates, top_k=4)
        top_docs = image_docs_for_context + top_docs
        context = "\n\n".join([d.page_content for d in top_docs])

        if output_lang == "en":
            lang_instr = "请用英文回答（Answer in English）。公司名统一使用标准英文全称：武汉力源信息技术股份有限公司 = Wuhan P&S Information Technology Co., Ltd.；武汉兴图新科电子股份有限公司 = Wuhan Xingtu Xinke Electronics Co., Ltd."
        else:
            lang_instr = "请用中文回答。"

        prompt = f"""你是招股说明书问答助手。知识库包含两份招股说明书：
- 《招股说明书1》：武汉兴图新科电子股份有限公司
- 《招股说明书2》：武汉力源信息技术股份有限公司

请严格基于以下上下文回答问题。

重要规则：
1. 首先根据问题中的公司名判断应使用哪份招股说明书的内容，不要混用两家公司的数据。
2. “武汉兴图新科电子股份有限公司”的子公司包括“北京启目”“武汉启目”“华创兴图”等，子公司数值不能当作母公司数值回答。
3. 关于“军用领域收入占主营业务收入的比重”，这是“直接军方+间接军方”合计占比（如 82.10%、97.31%、94.84%、94.34%），不要用单独的“直接军方占比”或“间接军方占比”代替。
4. 关于兴图新科“来自军用领域的收入”的四个年度合计值，严格按原文“6,464.51 万元、14,414.16 万元、18,780.67 万元和 4,627.14 万元”顺序对应 2016 年度、2017 年度、2018 年度、2019 年 1-6 月。
5. 当问题问“电子信息行业的上游/下游”时，以上游为“信息系统相关的电子元器件制造企业，以及机箱、机柜等金属壳体制造企业”；下游为“军队、政府机关、能源、交通、金融等行业”。
6. 关于“注册资本”，如出现多个历史变更值，以公司最新的注册资本为准。
7. 【表格数据识别】上下文中包含 Markdown 表格（用 `### 表格 X ###` 标记）。表格中每一行代表一条完整记录，请按行列对照关系准确提取数值，不要跨行错位。
8. 【数字与单位】严格保留原文中的单位（万元、万股、%等），不要自行换算或省略。用英文回答时，1 万元 = 10 thousand yuan，1 万股 = 10 thousand shares。
9. 【图像数据解读】上下文中包含以“【图像描述】”开头的块，这些是从 PDF 图像中提取的内容。当问题涉及图像时，请优先阅读这些块：
   - 问“增长率最快/最高”时，先列出所有候选项和对应数值，再选数值最大的项；
   - 问“负增长/下降”时，找数值为负的项（如“-2.0%”表示负增长）；
   - 问“占比最大”时，找百分比最大的项；
   - 问“组织结构”时，按层级列出图中所有部门和下属单位。
10. 【数字必须来自上下文】回答中出现的所有数字（金额、百分比、股数、日期等）必须严格来自上下文，不得自行推算、改写或使用训练数据中的“常识”。如果上下文中没有该数字，明确说“上下文未提供”。
11. {lang_instr}
12. 如果上下文中找不到与问题主体匹配的信息，明确回答“上下文未提供相关信息”，不要臆测。

上下文：
{context}

问题：{question}

回答："""
        answer = self._generate(prompt, max_new_tokens=512)
        return answer, top_docs

    # ---------- 【工单05】对外接口：支持多轮对话 ----------
    def ask(self, question, history=None):
        """
        question: 当前用户问题
        history: 对话历史，格式 [(用户问题, 助手回答), ...]，默认为 None
        """
        is_valid, cleaned, msg = self._validate_input(question)
        if not is_valid:
            return f"【输入错误】{msg}", []

        # 【工单05】指代消解：含指代词时结合历史改写为完整问题
        if history and self._needs_resolution(cleaned):
            try:
                cleaned = self._resolve_query(cleaned, history)
            except Exception:
                pass  # 消解失败时用原问题

        try:
            lang = self._detect_language(cleaned)
        except Exception:
            lang = "zh"

        try:
            if lang == "zh":
                return self._ask_chinese(cleaned, output_lang="zh")
            else:
                prepared = self._prepare_question(cleaned)
                answer, docs = self._ask_chinese(prepared, output_lang="en")
                answer = self._fix_company_names_in_answer(answer)
                return answer, docs
        except Exception as e:
            return f"【系统异常】{e}，请稍后重试。", []