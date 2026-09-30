# -*- coding: utf-8 -*-
"""
RAGAS 评测模块：对 RAG 系统的检索质量和生成质量做自动评估

四大核心指标：
    1. Faithfulness（忠实度）：回答是否忠于检索资料，有没有编造
    2. Answer Relevancy（答案相关性）：回答是否切题
    3. Context Precision（上下文精确度）：检索到的资料有多少是真正有用的
    4. Context Recall（上下文召回率）：回答所需的信息是否都检索到了

评测原理：
    RAGAS 不需要人工标注答案，它用 LLM 自己当裁判：
    - 把 (question, answer, contexts, ground_truth) 喂给评估 LLM
    - 评估 LLM 打分，输出 0-1 的分数
    - 4 个指标取平均就是综合分

评测需要：
    - 一个评估用 LLM（当裁判），用 DeepSeek API
    - 一组测试数据（问题 + 标准答案 + 检索到的上下文）

用法：
    H:\\an\\envs\\langchain2\\python.exe tests\\test_ragas.py

本模块在系统中的位置：
    这是「质量评估」工具，不参与线上请求。它可以直接用写死的样本（静态模式），
    也可以先用 TestClient 真实调用 main.py 的 /chat 与 /ingest 接口收集样本（动态模式）。
    下游依赖：config（DeepSeek 与本地向量模型配置）、logger；动态模式还会真实用到 main.py 的全链路。

指标怎么读（每个指标各管一段责任，出问题时要对着看）：
    - Context Recall 低 → 检索没召回全，先查切分粒度/召回条数/BM25 是否刷过；
    - Context Precision 低 → 召回了一堆无关内容，先查重排与 SCORE_THRESHOLD 阈值；
    - Faithfulness 低 → 模型在编，先看提示词约束和上下文是否被截断；
    - Answer Relevancy 低 → 检索没问题但答非所问，先查提示词与后处理。

关键取舍：
    - 评估用的 embedding 故意和主链路用同一个本地 bge-m3（不换成别的模型），
      这样 answer_relevancy 算出的语义相似度才和线上语义空间一致，分数才有参考意义；
      评估 LLM 则用 DeepSeek，temperature=0 让打分可复现。
    - 跑动态模式时把集合名切成临时集合 "_test_ragas"，评测结束再 drop，
      不污染正式知识库。
"""

import os  # 操作系统接口：读环境变量
import sys  # 系统接口：修改 sys.path
import json  # JSON 序列化

# 把项目根目录加入 sys.path（测试文件在 tests/ 下，要 import 项目模块）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 插入项目根目录

# 设置环境变量（用临时集合测试，不碰正式数据）
os.environ["MILVUS_COLLECTION"] = "_test_ragas"  # 临时集合名：必须在 import main/config 之前设置，config 在导入时就读取它
os.environ["MILVUS_URI"] = "http://localhost:19530"  # Milvus 地址

from langchain_openai import ChatOpenAI  # LangChain 的 OpenAI 兼容 LLM 封装
from langchain_huggingface import HuggingFaceEmbeddings  # 本地向量模型（RAGAS 评估器也需要 embedding）

from config import (  # 配置
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,  # DeepSeek 配置
    EMBED_MODEL_PATH, EMBED_DEVICE,  # 本地向量模型配置
)
from logger import get_logger  # 日志

# RAGAS 0.4.x 的导入路径（新 API）
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset  # 数据集结构
from ragas.llms import LangchainLLMWrapper  # LLM 包装器（把 LangChain LLM 适配给 RAGAS）
from ragas.embeddings import LangchainEmbeddingsWrapper  # Embedding 包装器
from ragas.metrics.collections import (  # 评估指标（0.4.x 从 collections 导入）
    faithfulness,  # 忠实度：回答是否编造了资料里没有的内容
    answer_relevancy,  # 答案相关性：回答是否切题
    context_precision,  # 上下文精确度：检索到的资料有多少是真正有用的
    context_recall,  # 上下文召回率：回答所需信息是否都检索到了
)
from ragas import evaluate  # RAGAS 评测主函数

logger = get_logger(__name__)  # 本模块 logger


# ==================== 1. 评测 LLM 初始化 ====================

def get_evaluator_llm() -> LangchainLLMWrapper:
    """
    构建评估 LLM：用 DeepSeek 当裁判
    评估 LLM 和被评估的 RAG LLM 可以是同一个（成本低）也可以不同（更客观）

    返回：LangchainLLMWrapper —— 把 LangChain 的 ChatOpenAI 包成 RAGAS 能调用的 LLM。
    说明：temperature=0 是为了打分稳定可复现（评测不是创作，不需要随机性）；
          max_tokens 给到 4096 是因为 RAGAS 要求模型先写推理过程再给分，输出会比较长。
    异常：API Key 无效、网络不通时不会在这里报错，而是到 evaluate() 阶段才抛异常，
          由调用方捕获后打印错误。
    """
    # 用 DeepSeek 作为评估裁判
    eval_llm = ChatOpenAI(  # LangChain LLM（OpenAI 兼容接口）
        model=LLM_MODEL,  # 模型名 "deepseek-chat"
        api_key=LLM_API_KEY,  # API Key
        base_url=LLM_BASE_URL,  # 接口地址
        temperature=0,  # 评估时温度设 0（确定性输出，打分稳定）
        max_tokens=4096,  # 评估需要长输出（分析理由 + 打分）
    )
    # 用 LangchainLLMWrapper 包装：让 RAGAS 能调用这个 LLM
    wrapper = LangchainLLMWrapper(eval_llm)  # 包装
    logger.info(f"RAGAS 评估 LLM 就绪：{LLM_MODEL} @ {LLM_BASE_URL}")
    return wrapper  # 返回包装器


def get_evaluator_embeddings() -> LangchainEmbeddingsWrapper:
    """
    构建 Embedding 包装器：RAGAS 的 answer_relevancy 指标需要 embedding 做语义相似度
    用和 RAG 系统同一个 bge-m3（保持语义空间一致）

    返回：LangchainEmbeddingsWrapper —— 包好后的 embedding，供 RAGAS 内部算相似度。
    说明：
        - model_kwargs 指定跑在 EMBED_DEVICE（这里是 CPU）：评测机器一般没有 GPU，
          而且评测是离线跑，慢一点可以接受；
        - normalize_embeddings=True 与主链路保持一致，配合 COSINE 度量算出的相似度才可比。
    注意：这里会重新加载一份向量模型（和 main.py 里的单例不是同一个实例），
          所以同时跑服务和评测会额外占用一份内存。
    """
    embeddings = HuggingFaceEmbeddings(  # 本地 bge-m3，不联网
        model_name=EMBED_MODEL_PATH,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
    wrapper = LangchainEmbeddingsWrapper(embeddings)  # 包装
    logger.info(f"RAGAS 评估 Embedding 就绪：{EMBED_MODEL_PATH}")
    return wrapper  # 返回包装器


# ==================== 2. 测试数据集 ====================
# 每组测试数据包含：
#   user_input    : 用户的提问
#   response      : RAG 系统生成的回答（被评估对象）
#   retrieved_contexts : RAG 检索到的上下文片段列表
#   reference     : 标准答案（ground truth，人工标注或从原文取）
#
# 实际使用时：先调 /chat 接口拿 response 和 retrieved_contexts，再填 reference
#
# 静态模式下这几个用例是人工写死的，好处是不依赖线上服务、结果可复现；
# 代价是与真实检索结果可能脱节，所以要定期用动态模式（dynamic_evaluation）复核。
# 第 3 条用例是刻意构造的「负面样本」：检索到的资料和问题完全无关，回答却写得很像样，
# 用来验证 faithfulness 能不能把「编造」识别出来——如果三项分数都接近 1，说明评测没生效。

TEST_DATASET = [
    {
        # 测试用例 1：法律咨询 - 相邻关系
        "user_input": "邻居半夜装修噪音很大，我可以主张什么权利？",  # 用户问题
        "retrieved_contexts": [  # 检索到的资料（模拟 RAG 的检索结果）
            "不动产的相邻权利人应当按照有利生产、方便生活、团结互助、公平合理的原则，正确处理相邻关系。",
            "不动产权利人不得违反国家规定弃置固体废物，排放大气污染物、水污染物、土壤污染物、噪声、光辐射、电磁辐射等有害物质。",
            "（2021）京0105民初12345号：业主长期在夜间进行高噪声装修，法院判令停止侵害并赔偿精神损害抚慰金2000元。",
        ],
        "response": (  # RAG 生成的回答（被评估）
            "根据《民法典》第二百八十八条和第二百九十四条，不动产权利人不得排放噪声等有害物质。"
            "邻居半夜装修属于噪音扰民，您可以主张停止侵害和损害赔偿。"
            "参考判例（2021）京0105民初12345号，法院判令停止侵害并赔偿精神损害抚慰金2000元。"
            "建议先与邻居协商，协商不成可向物业投诉或向法院起诉。"
        ),
        "reference": (  # 标准答案（人工标注，用于 context_recall 评估）
            "根据民法典相邻关系的规定，不动产权利人不得排放噪声。"
            "邻居夜间高噪声装修属于侵权行为，可主张停止侵害和赔偿损失。"
            "已有判例支持精神损害抚慰金的赔偿请求。"
        ),
    },
    {
        # 测试用例 2：法律咨询 - 合同违约
        "user_input": "房东不退押金怎么办？",  # 用户问题
        "retrieved_contexts": [  # 检索到的资料
            "租赁合同解除后，承租人应当返还租赁物。出租人应当返还押金，但可以扣除承租人未支付的租金或违约金。",
            "当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。",
        ],
        "response": (  # RAG 回答
            "房东不退押金属于违约行为。根据法律规定，租赁合同解除后出租人应当返还押金，"
            "但可以扣除未支付的租金或违约金。如果房东无正当理由拒不退还，"
            "您可以向法院起诉要求返还押金并承担违约责任。"
        ),
        "reference": (  # 标准答案
            "租赁合同解除后出租人应返还押金，可扣除欠租或违约金。"
            "房东无理由不退押金属于违约，租客可起诉要求返还。"
        ),
    },
    {
        # 测试用例 3：测试编造检测（Faithfulness 应该扣分）
        "user_input": "什么是知识产权？",  # 用户问题
        "retrieved_contexts": [  # 检索到的资料（只有相邻关系的内容，和问题不相关）
            "不动产的相邻权利人应当按照有利生产、方便生活、团结互助、公平合理的原则，正确处理相邻关系。",
        ],
        "response": (  # RAG 回答（故意编造了一些内容，测试 Faithfulness 能否检测到）
            "知识产权是指对智力成果享有的专有权利，包括专利权、商标权、著作权等。"
            "根据《民法典》第一百二十三条，知识产权包括作品、发明、实用新型、外观设计、商标等客体。"
            "知识产权具有专属性、地域性和时间性。"
        ),
        "reference": "知识产权是对智力成果的专有权利，包括专利、商标、著作权等。"
    },
]


# ==================== 3. 构建评测数据集 ====================

def build_eval_dataset() -> EvaluationDataset:
    """
    把 TEST_DATASET 列表转成 RAGAS 的 EvaluationDataset 格式

    SingleTurnSample 是 RAGAS 0.4.x 的单轮对话评测样本结构：
        user_input           : 用户输入
        response             : 被评估的回答
        retrieved_contexts   : 检索到的上下文列表
        reference            : 标准答案（ground truth）

    返回：EvaluationDataset —— RAGAS 要求的评测集容器。
    说明：四个字段缺一不可——context_recall 必须有 reference 才能算，
          context_precision 依赖 retrieved_contexts，faithfulness 依赖 response 与 contexts 的对照。
    """
    samples = []  # 存放 SingleTurnSample
    for item in TEST_DATASET:  # 遍历测试数据
        sample = SingleTurnSample(  # 构建单个评测样本
            user_input=item["user_input"],  # 用户问题
            response=item["response"],  # RAG 回答
            retrieved_contexts=item["retrieved_contexts"],  # 检索上下文
            reference=item["reference"],  # 标准答案
        )
        samples.append(sample)  # 加入列表
    dataset = EvaluationDataset(samples=samples)  # 包装成 EvaluationDataset
    logger.info(f"评测数据集已构建：{len(samples)} 条测试用例")
    return dataset  # 返回数据集


# ==================== 4. 执行评测 ====================

def run_evaluation():
    """
    执行 RAGAS 评测，返回四大指标分数

    流程：
        1. 构建评估 LLM + Embedding
        2. 构建评测数据集
        3. 调用 ragas.evaluate() 跑四大指标
        4. 打印每条用例的详细得分 + 总体平均分

    返回：ragas.evaluate() 的结果对象（可用 to_pandas() 转成 DataFrame）；
          评测抛异常时返回 None，不往上抛（这是离线工具，失败只提示）。
    异常与降级：网络/额度/超时导致的评测失败会被就地捕获并打印，
                临时结果文件也不会生成，不会留下半截脏数据。
    说明：评测会真实调用 DeepSeek，一条用例要跑多次 LLM 调用，整体通常需要 1~3 分钟。
    """
    logger.info("=" * 60)
    logger.info("开始 RAGAS 评测 ...")
    logger.info("=" * 60)

    # 1. 构建评估器
    evaluator_llm = get_evaluator_llm()  # 评估 LLM（当裁判）
    evaluator_emb = get_evaluator_embeddings()  # 评估 Embedding

    # 2. 构建评测数据集
    dataset = build_eval_dataset()  # 评测数据集

    # 3. 定义要跑的指标
    metrics = [  # 四大核心指标
        faithfulness,  # 忠实度：回答有没有编造资料里没有的内容
        answer_relevancy,  # 答案相关性：回答是否切题
        context_precision,  # 上下文精确度：检索到的资料有多少是真正有用的
        context_recall,  # 上下文召回率：回答所需信息是否都检索到了
    ]

    # 4. 执行评测
    logger.info("正在执行评测（会调用 LLM 打分，可能需要 1-3 分钟）...")
    try:  # 评测可能因网络/超时失败
        result = evaluate(  # RAGAS 评测主函数
            dataset=dataset,  # 评测数据集
            metrics=metrics,  # 指标列表
            llm=evaluator_llm,  # 评估 LLM
            embeddings=evaluator_emb,  # 评估 Embedding
        )
    except Exception as e:  # 捕获异常
        logger.error(f"评测失败：{e}")
        print(f"\n[错误] 评测失败：{e}")
        return None  # 返回 None 而不是让异常冒泡：便于上层（__main__）继续走完流程

    # 5. 打印结果
    print("\n" + "=" * 60)
    print("RAGAS 评测结果")
    print("=" * 60)

    # result 是一个 DataFrame（每行一条用例，每列一个指标）
    results_df = result.to_pandas()  # 转 Pandas DataFrame 方便查看

    # 逐条打印
    for i, row in results_df.iterrows():  # 遍历每条用例
        print(f"\n--- 用例 {i + 1} ---")
        print(f"  问题：{row.get('user_input', '')[:60]}...")  # 用 get 取值：不同 RAGAS 版本列名可能缺失，缺了就显示 N/A
        print(f"  Faithfulness（忠实度）：{row.get('faithfulness', 'N/A')}")  # 忠实度
        print(f"  Answer Relevancy（答案相关性）：{row.get('answer_relevancy', 'N/A')}")  # 相关性
        print(f"  Context Precision（上下文精确度）：{row.get('context_precision', 'N/A')}")  # 精确度
        print(f"  Context Recall（上下文召回率）：{row.get('context_recall', 'N/A')}")  # 召回率

    # 总体平均分
    print("\n" + "-" * 60)
    print("总体平均分：")
    print(f"  Faithfulness（忠实度）       : {results_df['faithfulness'].mean():.4f}")  # 忠实度均值
    print(f"  Answer Relevancy（答案相关性）: {results_df['answer_relevancy'].mean():.4f}")  # 相关性均值
    print(f"  Context Precision（上下文精确）: {results_df['context_precision'].mean():.4f}")  # 精确度均值
    print(f"  Context Recall（上下文召回）   : {results_df['context_recall'].mean():.4f}")  # 召回率均值
    overall = results_df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean().mean()
    print(f"  综合                          : {overall:.4f}")  # 四项均值：先对列求均值再对四项求均值，等价于所有单元格的平均
    print("=" * 60)

    # 保存结果到 JSON
    output_path = os.path.join(os.path.dirname(__file__), "..", "ragas_result.json")  # 结果文件路径：写到项目根目录，方便和代码一起看
    results_df.to_json(output_path, orient="records", force_ascii=False, indent=2)  # 写 JSON：records 便于逐条阅读，force_ascii=False 保留中文
    print(f"\n详细结果已保存到：{output_path}")

    return result  # 返回评测结果


# ==================== 5. 动态评测（真实调用 RAG 系统） ====================

def dynamic_evaluation(num_cases: int = 3):
    """
    动态评测：真实调用 /chat 接口拿回答，再用 RAGAS 打分

    与静态评测的区别：
        - 静态评测（run_evaluation）：回答和上下文都是预先写好的测试数据
        - 动态评测：真实调 RAG 系统拿回答和上下文，再填标准答案做评测

    流程：
        1. 往 Milvus 入测试数据（民法典法条 + 判例）
        2. 用 TestClient 调 /chat 接口拿 response + sources
        3. 填上人工标准答案
        4. 用 RAGAS 打分

    参数：num_cases 最多评几条（默认 3）——每条都要真实跑一次 /chat + RAGAS 的多次裁判调用，
          很费时间和额度，所以限制条数。
    返回：ragas.evaluate() 的结果对象；没有收集到任何样本时返回 None。
    异常与降级：单条 /chat 调用失败只记日志并跳过（continue），不影响其它用例；
                正常跑完后会 drop 掉临时集合 _test_ragas；若 evaluate() 中途抛异常，
                这一步清理会被跳过，由下次运行开头的预清理兜底。
    说明：只有 reference（标准答案）是人工写的，其余都来自真实系统，所以这个模式最能反映线上质量；
          但成本高、结果会随语料变化而波动，适合在改完检索参数后跑一次做前后对比。
    """
    from fastapi.testclient import TestClient
    from pymilvus import MilvusClient

    # 导入 main 会触发 lifespan（加载模型）
    from main import app

    # 清理旧测试数据
    mc = MilvusClient(uri="http://localhost:19530")
    if mc.has_collection("_test_ragas"):
        mc.drop_collection("_test_ragas")  # 先删掉上一次残留的临时集合，保证本次评测从干净状态开始

    tc = TestClient(app)  # 创建测试客户端

    # 1. 入库测试数据
    legal_text = """# 民法典 相邻关系

## 第二百八十八条
不动产的相邻权利人应当按照有利生产、方便生活、团结互助、公平合理的原则，正确处理相邻关系。

## 第二百九十四条
不动产权利人不得违反国家规定弃置固体废物，排放大气污染物、水污染物、土壤污染物、噪声、光辐射、电磁辐射等有害物质。

# 判例摘要
（2021）京0105民初12345号：业主长期在夜间进行高噪声装修，法院判令停止侵害并赔偿精神损害抚慰金2000元。
"""
    tc.post("/ingest/text", json={"source": "民法典测试.md", "text": legal_text, "strategy": "paragraph"})  # 入库后接口内部会刷新 BM25，保证检索两路都能命中

    # 2. 获取角色 ID
    roles_resp = tc.get("/roles")
    role_id = roles_resp.json()["roles"][0]["id"]  # 第一个角色（法律顾问）：动态评测只针对这一个角色

    # 3. 定义测试问题 + 标准答案
    eval_cases = [
        {
            "query": "邻居半夜装修噪音很大，我可以主张什么权利？",  # 测试问题
            "reference": "根据民法典相邻关系规定，可主张停止侵害和赔偿损失，已有判例支持精神损害抚慰金。",
        },
        {
            "query": "相邻关系的处理原则是什么？",  # 测试问题
            "reference": "有利生产、方便生活、团结互助、公平合理是处理相邻关系的四项原则。",
        },
        {
            "query": "民法典对排放噪声有什么规定？",  # 测试问题
            "reference": "不动产权利人不得违反国家规定排放噪声等有害物质。",
        },
    ]

    # 4. 逐条调 /chat 接口拿回答和上下文
    samples = []  # RAGAS 样本列表
    for case in eval_cases[:num_cases]:  # 只取前 num_cases 条
        r = tc.post("/chat", json={  # 调 /chat
            "user_id": 999,  # 测试用户：用不常出现的 ID，避免和真实用户的历史记忆混在一起
            "role_id": role_id,  # 角色ID
            "query": case["query"],  # 查询
            "top_k": 3,  # 引用条数
        })
        if r.status_code != 200:  # 调用失败
            logger.error(f"调用 /chat 失败：{r.text}")
            continue  # 跳过这一条，继续收集后面的用例
        data = r.json()  # 解析响应
        response = data["answer"]  # RAG 回答
        contexts = [s["content"] for s in data["sources"]]  # 检索到的上下文：只取接口返回的 sources 文本

        sample = SingleTurnSample(  # 构建 RAGAS 样本
            user_input=case["query"],
            response=response,
            retrieved_contexts=contexts,
            reference=case["reference"],
        )
        samples.append(sample)
        logger.info(f"用例收集：Q={case['query'][:30]}... | 检索到 {len(contexts)} 条上下文")

    # 5. 执行评测
    if not samples:  # 没有样本
        print("[错误] 没有收集到评测样本")
        return None  # 一条都没收集到就没必要调 RAGAS 了

    dataset = EvaluationDataset(samples=samples)  # 构建数据集
    evaluator_llm = get_evaluator_llm()  # 评估 LLM
    evaluator_emb = get_evaluator_embeddings()  # 评估 Embedding

    print("\n" + "=" * 60)
    print("RAGAS 动态评测（真实调用 RAG 系统）")
    print("=" * 60)

    result = evaluate(  # 执行评测
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_emb,
    )

    # 打印结果
    results_df = result.to_pandas()
    for i, row in results_df.iterrows():
        print(f"\n--- 用例 {i + 1} ---")
        print(f"  问题：{row.get('user_input', '')[:60]}...")
        print(f"  Faithfulness（忠实度）       : {row.get('faithfulness', 'N/A')}")
        print(f"  Answer Relevancy（答案相关性）: {row.get('answer_relevancy', 'N/A')}")
        print(f"  Context Precision（上下文精确）: {row.get('context_precision', 'N/A')}")
        print(f"  Context Recall（上下文召回）   : {row.get('context_recall', 'N/A')}")

    print("\n" + "-" * 60)
    print("总体平均分：")
    for col in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
        if col in results_df.columns:  # 先判断列存在再求均值：缺列的 RAGAS 版本直接跳过，避免 KeyError
            print(f"  {col:30s}: {results_df[col].mean():.4f}")
    print("=" * 60)

    # 清理
    mc.drop_collection("_test_ragas")  # 评测用的临时集合用完即删，不留垃圾数据

    return result


# ==================== 主入口 ====================
if __name__ == "__main__":
    import argparse  # 命令行参数解析

    parser = argparse.ArgumentParser(description="RAGAS 评测工具")  # 创建解析器
    parser.add_argument(  # 添加参数
        "--mode",  # 参数名
        choices=["static", "dynamic"],  # 可选值
        default="static",  # 默认静态评测
        help="评测模式：static=静态评测（预设数据）；dynamic=动态评测（真实调用RAG系统）",
    )
    args = parser.parse_args()  # 解析参数

    if args.mode == "dynamic":  # 动态模式
        print("[模式] 动态评测：真实调用 RAG 系统拿回答再评测")
        dynamic_evaluation(num_cases=3)  # 执行动态评测
    else:  # 静态模式
        print("[模式] 静态评测：使用预设的测试数据集")
        run_evaluation()  # 执行静态评测
