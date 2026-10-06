# -*- coding: utf-8 -*-
"""RAGAS 评测：参考答案摘自指南原文；急症路径用规则测试不走 RAGAS。python scripts/05_ragas_eval.py"""
import os  # 标准库：取脚本所在目录，向上拼项目根路径
import sys  # 标准库：把项目根插入模块搜索路径，脚本才能 import src
from pathlib import Path  # 面向对象路径：拼输出 CSV 的跨平台路径

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # 本文件在 scripts/ 下，向上两级即项目根；插到最前确保优先命中

import pandas as pd  # 数据表：评测结果组 DataFrame、写 CSV 都靠它
from ragas import evaluate, EvaluationDataset  # RAGAS 框架入口：evaluate 跑指标评分，EvaluationDataset 封装样本集
from ragas.dataset_schema import SingleTurnSample  # 单轮样本结构：问题/参考答案/生成答案/检索上下文四要素
from ragas.llms import LangchainLLMWrapper  # 把 LangChain 的 ChatOpenAI 适配成 RAGAS 认的 Judge 接口
from ragas.run_config import RunConfig  # 评分运行配置：超时/重试/并发，防 DeepSeek 限流
from ragas.metrics import (Faithfulness, LLMContextRecall,  # 忠实度（生成是否忠于检索内容）、上下文召回率
                           LLMContextPrecisionWithReference, ResponseRelevancy)  # 带参考的上下文精度、答案相关性
from langchain_openai import ChatOpenAI  # DeepSeek 在线 API 客户端，此处充当评分裁判（Judge）

from src.config import settings  # 统一配置中心：模型名、API key、embedding 模型都从 .env 读

ROLE_ID = "doctor"  # 固定用医生角色跑评测：与线上主场景人设一致，分数才有代表性

# 标准问答集：reference 严格按检索命中的原文表述摘录（知识库按 55 字小 chunk 切块）
TEST_DATA = [  # 4 道高频题：覆盖数值/清单/口诀/建议四种问法
    {"user_input": "高血压患者每天吃盐不能超过多少克？",  # 题1：限盐数值题——答案短而精确，考检索命中
     "reference": "每人每日食盐摄入量不超过5克，推荐使用带刻度的盐勺；注意隐性盐的摄入。"},  # 参考答案必须是原文摘录而非改写，否则忠实度评估失准
    {"user_input": "高血压常用的降压药物有哪些类别？",  # 题2：药物分类清单题——考多 chunk 召回完整性
     "reference": "基层医疗卫生机构应配备下述五大类降压药，分别以A、B、C、D简称："  # 原文跨多个 55 字小 chunk，隐式拼接还原完整表述
                  "A为ACEI和ARB，C为钙拮抗剂（CCB），D为噻嗪类利尿剂，"  # 五大类缩写是高频考点，检索必须命中这些术语
                  "并配备由上述五大类药物组成的单片复方制剂。"},  # 续完：整段与指南原文一致
    {"user_input": "健康生活方式八部曲的内容是什么？",  # 题3：口诀型长答案题——考长上下文忠实度
     "reference": "限盐减重多运动，戒烟戒酒心态平，营养平衡睡得香。合理膳食、减少钠盐及增加钾盐"  # 八部曲口诀原文
                  "摄入、减轻体重、规律的中等强度运动均有直接降压效果；减轻精神压力、保证好的睡眠"  # 口诀对应的展开解释
                  "也是提高治疗效果的重要方面。"},  # 续完
    {"user_input": "高血压患者适合做哪些运动？",  # 题4：运动建议题——与题3部分原文重叠，验证指标区分度
     "reference": "规律的中等强度运动（如快走、慢跑、骑车、游泳、太极拳等）均有直接的降压效果。"},  # 单 chunk 即可覆盖的短参考
]

# 四指标：召回/精度评检索质量，忠实度/相关性评生成质量（事实正确性与原文参考答案同源，不评）
_CN_COLUMNS = [("recall", "上下文召回率"), ("precision", "上下文精度"),  # 指标名子串→中文列名映射
               ("faithful", "忠实度"), ("relevan", "答案相关性")]  # ragas 指标名带版本后缀，故用子串模糊匹配


def _build_components():  # 加载与线上一致的 RAG 组件：评测必须走真实链路，分数才代表线上
    """加载与线上一致的 RAG 组件（embedder/BM25/角色/主链）。"""
    from src.bm25_index import BM25Index  # BM25 关键词倒排索引（函数内导入：重资源按需加载，脚本启动快）
    from src.ingestion import get_embedder  # 加载 bge 向量模型（首次需从 HuggingFace 下载权重，较慢）
    from src.rag_chain import RAGChain  # 与线上一致的 RAG 主链，即被评测对象
    from src.retrieval import load_records  # 读知识库 chunk 记录，作 BM25 语料
    from src.role import RoleManager  # 角色人设管理：全科/西医/心理

    embedder = get_embedder()  # 向量模型单例：文本转向量供语义检索
    bm25 = BM25Index(); bm25.build(load_records())  # 建 BM25 索引：分词→统计词频/IDF，供字面精确检索
    return embedder, bm25, RAGChain(RoleManager())  # 返回检索两路组件 + 注入角色的 RAG 主链


def _run_one(chain, embedder, bm25, question):  # 跑单题：走完整 RAG 链路拿答案与检索上下文
    resp = chain.prepare(ROLE_ID, question, embedder, bm25)  # prepare 只做检索与决策不生成：拿到模式/提示词/引用
    if resp.mode == "answer":  # 正常命中知识库
        answer = "".join(chain.stream_answer(ROLE_ID, resp.prompt))  # 流式生成的全部 chunk 拼成完整答案
    elif resp.mode == "no_content":  # 未命中：走 LLM 通用知识兜底
        answer = chain.general_tips(ROLE_ID, resp.query)  # 兜底非流式一次性返回
    else:  # clarify / emergency 直接用系统话术，不作为知识问答评分主体
        answer = resp.message  # 固定话术直接当答案
    return answer.strip(), [c["text"] for c in resp.citations]  # 答案 + 检索片段原文列表（RAGAS 的 retrieved_contexts）


def _check_emergency(chain, embedder, bm25):  # 急症场景不进 RAGAS：固定文案无生成过程，LLM 评分无意义，规则断言更可靠
    """急症路径规则测试（不走 RAGAS）：断言固定文案包含 120 与急症提示。"""
    resp = chain.prepare(ROLE_ID, "胸痛", embedder, bm25)  # "胸痛"是急症关键词，应触发安全层拦截
    ok = resp.mode == "emergency" and "120" in resp.message  # 双重断言：模式必须是 emergency 且文案含 120 急救电话
    print(f"🚨 急症规则测试（RAGAS 评测集外）：{'✅ 通过' if ok else '❌ 未通过'}")  # 即时可见：急症拦截失效要第一时间发现


def _embeddings_wrapper():  # 构造 RAGAS 侧答案相关性指标所需的 embedding 包装器
    # ResponseRelevancy 需要 embedding；失败则降级为不跑该指标
    try:  # 本地向量模型加载可能失败（无网络/缺依赖），做好降级
        from langchain_community.embeddings import HuggingFaceEmbeddings  # 本地 HuggingFace 向量模型（与入库同款 bge）
        from ragas.embeddings import LangchainEmbeddingsWrapper  # 适配成 RAGAS 认的 embedding 接口
        return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=settings.embedding_model))  # 与入库一致的向量模型，保证相关性打分与检索同分布
    except Exception as exc:  # 加载失败不致命
        print(f"⚠️ embedding 包装不可用，跳过答案相关性：{exc}")  # 明确提示哪项指标被跳过
        return None  # 返回 None，主流程据此不挂 ResponseRelevancy


def _save_csv(samples, scores, errors, out: Path) -> None:  # 评分结果落盘为中文 CSV 明细
    """逐指标独立评分后的结果重组为中文列名明细；失败格子写明真实异常原因。"""
    rows = []  # 每行对应一道题
    for i, s in enumerate(samples):  # 逐题组装
        row, failed = {"问题": s.user_input}, []  # 首列是问题；failed 收集该题失败指标及原因
        for key, cn in _CN_COLUMNS:  # 按固定顺序遍历四指标
            mname = next((k for k in scores if key in k.lower()), None)  # 子串模糊匹配真实指标名（ragas 不同版本指标名略有差异）
            if mname is None:  # 该指标整体未跑（如 embedding 不可用）
                continue  # 跳过不占列
            v = scores[mname][i]  # 取该题该指标的分数
            if v is None or (isinstance(v, float) and pd.isna(v)):  # None=评分失败；NaN=ragas 返回空值
                row[cn] = "未计算"  # 失败格子显式标"未计算"，区别于真实的 0 分
                failed.append(f"{cn}（{errors.get(mname, 'Judge 未返回')}）")  # 备注写明真实异常原因，便于排查
            else:  # 正常出分
                row[cn] = round(float(v), 3)  # 保留 3 位小数，报告整洁
        row["备注"] = "原文摘录 · 高血压防治指南2025版" + (f"；{'; '.join(failed)}" if failed else "")  # 备注固定标注参考来源，有失败时追加原因
        rows.append(row)  # 收进结果集
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")  # utf-8-sig 带 BOM：Excel 打开中文不乱码


def main():  # 主流程：加载组件→跑题→逐指标评分→落盘
    print("⏳ 加载 RAG 组件...")  # 向量模型加载较慢，先给提示
    embedder, bm25, chain = _build_components()  # 与线上一致的组件，无旁路

    samples = []  # 收集 RAGAS 样本
    for i, item in enumerate(TEST_DATA, 1):  # 逐题跑 RAG，题号从 1 起便于阅读
        answer, contexts = _run_one(chain, embedder, bm25, item["user_input"])  # 完整链路拿答案与上下文
        print(f"  [{i}/{len(TEST_DATA)}] {item['user_input']} → {len(contexts)} 段 / {len(answer)} 字｜{answer[:40]}")  # 打印进度与答案预览，跑批时心里有数
        samples.append(SingleTurnSample(  # 组装单轮样本四要素
            user_input=item["user_input"], reference=item["reference"],  # 问题 + 金标准参考答案
            response=answer or "（未生成回答）",  # 空答案兜底占位：ragas 不接受空串
            retrieved_contexts=contexts or ["（未检索到内容）"]))  # 空上下文同理占位，保证结构完整
    _check_emergency(chain, embedder, bm25)  # 急症规则断言：不进 RAGAS，在评测集外单独验证

    judge = LangchainLLMWrapper(ChatOpenAI(  # Judge 用 DeepSeek：评分本质是 LLM 判别任务
        model=settings.deepseek_model, api_key=settings.deepseek_api_key,  # 与线上一致的模型配置
        base_url=settings.deepseek_base_url, temperature=0, timeout=120, max_retries=3))  # temperature=0 评分求稳；timeout=120 防长评卡死；重试 3 次抗限流
    emb = _embeddings_wrapper()  # 答案相关性指标需要的 embedding，加载失败则为 None
    metrics = [LLMContextRecall(), LLMContextPrecisionWithReference(), Faithfulness()]  # 先挂三个不依赖 embedding 的指标：召回/精度/忠实度
    # strictness=1 强制候选问题 n=1，DeepSeek 不支持 n>1（否则 400 Invalid n value）
    if emb is not None: metrics.append(ResponseRelevancy(strictness=1))  # embedding 可用才挂相关性指标
    # max_workers=2 降并发避免 DeepSeek 限流；ragas 侧重试 3 次
    run_cfg = RunConfig(timeout=120, max_retries=3, max_wait=180, max_workers=2)  # 单请求 120s、重试 3 次、最长等 180s、并发压到 2
    dataset = EvaluationDataset(samples=samples)  # 样本集封装成 ragas 评测数据集

    scores, errors = {}, {}  # scores 存各指标分数列表，errors 存失败原因
    for m in metrics:  # 逐指标独立评分：单指标失败不拖垮其他指标，且能拿到真实异常
        print(f"⏳ 指标评分：{m.name} ...")  # 打印当前指标名，卡住时知道卡在哪个
        try:  # 本指标一切异常都兜住，不影响后续指标
            r = evaluate(dataset, metrics=[m], llm=judge, embeddings=emb,  # 单次只评一个指标
                         run_config=run_cfg, raise_exceptions=True)  # raise_exceptions=True：拿真实异常而非默认 NaN
            scores[m.name] = r.to_pandas()[m.name].tolist()  # 结果转 DataFrame 再取该指标列成分数列表
        except Exception as exc:  # 评分失败
            reason = f"{type(exc).__name__}: {str(exc)[:60]}"  # 异常类型 + 前 60 字符：够定位又不刷屏
            print(f"  ❌ {m.name} 计算失败：{reason}")  # 失败即时可见
            scores[m.name], errors[m.name] = [None] * len(samples), reason  # 该指标整列置 None，原因写进备注

    out = Path(__file__).resolve().parent.parent / "data" / "ragas_report.csv"  # 报告固定输出到项目根 data/ 下
    _save_csv(samples, scores, errors, out)  # 组表落盘
    print(f"\n📄 中文明细已保存：{out}（共 {len(metrics)} 指标 × {len(samples)} 题）")  # 总结输出路径与规模


if __name__ == "__main__":  # 脚本直跑入口（被 import 时不触发）
    main()  # 开跑
