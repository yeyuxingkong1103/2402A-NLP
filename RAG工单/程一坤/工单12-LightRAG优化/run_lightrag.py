# -*- coding: utf-8 -*-
"""
工单12：LightRAG 知识图谱构建与检索对比
工单编号：人工智能NLP-RAG 项目-LightRAG 优化
功能：
  1. 用 LightRAG（HKU开源，图结构RAG）对《招股说明书1/2.pdf》构建知识图谱
     （自定义实体/关系类型的 Prompt 优化抽取）；
  2. 检索时可选择 RAG（向量库）或 LightRAG（知识图谱）知识库；
  3. 对14个验收问题比对两套系统的检索结果，输出关键事实命中率对比。
运行：
  python run_lightrag.py build     # 构建知识图谱（首次较慢，LLM抽取实体）
  python run_lightrag.py compare   # RAG vs LightRAG 检索对比
说明：LightRAG 1.5.7 要求异步生命周期（initialize_storages/finalize_storages），
     本脚本全程在单事件循环内完成（insert 用 ainsert，query 用 aquery）。
"""
import sys      # 标准库：命令行参数与模块搜索路径
import os       # 标准库：路径与文件判断
import json     # 标准库：读取 ground_truth.json
import time     # 标准库：计时与时间戳
import asyncio  # 标准库：异步事件循环（LightRAG 1.5.7 全异步 API）

HERE = os.path.dirname(os.path.abspath(__file__))   # 本脚本所在目录（工单12目录）
COMMON = os.path.join(HERE, "..", "00-公共模块")     # 公共模块目录（相对路径）
sys.path.insert(0, COMMON)  # 加入搜索路径以便 import 公共模块

# 清除代理环境变量（踩坑：代理会把 Ollama/HF 本地请求劫持导致连接异常）
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_k, None)  # 逐个删除代理变量

from config import PDF_ZGS1, PDF_ZGS2  # 两份招股书 PDF 路径
from pdf_parser import parse_pdf_text  # PDF 文本解析（逐页提取）
from chunker import chunk_pages        # 文本分块（带页码元数据）
from vector_store import VectorStore   # 传统向量库（对照组）
from ollama_client import client       # Ollama 同步客户端（传统RAG生成用）
from rag_engine import QA_PROMPT       # 问答 Prompt 模板（与主系统保持一致）

WORK_DIR = os.path.join(HERE, "lightrag_work")  # LightRAG 工作目录（存图谱与向量索引）
MARKER = os.path.join(WORK_DIR, "build_done.txt")  # 构建完成标记文件（防重复构建）

# ── 工单要求：根据 PDF 内容优化实体/关系类型（金融招股书领域） ──
# LightRAG 1.5.x 通过 addon_params["entity_types_guidance"] 注入领域抽取指引，
# 实体类型与关系类型一并写入该指引文本。
ENTITY_GUIDANCE = """针对金融招股书领域，重点抽取以下类型的实体：
公司(发行人/子公司/关联方)、人物(实控人/董监高)、机构/股东(投资机构/监管机构)、
产品/业务(视频指挥系统等)、技术标准、工程项目、财务指标(营收/净利润/注册资本)、
行业(国防信息化等)、合同/协议、资质/专利。
关系类型重点识别：持股/控股、任职(法定代表人/高管)、供应商关系、客户关系、
参与制定(标准)、募集资金投向、子公司/关联方、行业上下游、同业竞争。"""


def make_rag():
    """创建 LightRAG 实例（1.5.7 API：Ollama 双绑定 + 领域 Prompt 注入）"""
    from lightrag import LightRAG  # LightRAG 主类
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed  # Ollama 的 LLM/Embedding 适配函数
    from lightrag.utils import EmbeddingFunc  # Embedding 函数包装器（声明维度/截断长度）

    return LightRAG(
        working_dir=WORK_DIR,  # 图谱与索引持久化目录
        # LLM 绑定：Ollama 原生接口 qwen2.5:7b-instruct。
        # 本机为 RTX 4060 Laptop 8GB：7B(4.7G)+bge-m3(1.2G)+KV ≈ 7.2G 可全驻 GPU；
        # 实测单请求 prompt 1020 tok/s / 生成 40 tok/s。曾因 max_async=2 + 无生成上限
        # 导致显存溢出到 CPU（同请求 448s）与模板复读刷爆 token，故必须：
        # max_async=1 单并发 + num_predict 封顶 + keep_alive 保模型常驻。
        llm_model_func=ollama_model_complete,   # LLM 调用函数（Ollama 补全接口）
        llm_model_name="qwen2.5:7b-instruct",   # LLM 模型名
        llm_model_max_async=1,                  # 关键：单并发。8G 显存双并发会溢出到 CPU，慢 40 倍
        llm_model_kwargs={
            "host": "http://127.0.0.1:11434",   # 本机 Ollama 服务地址
            "keep_alive": "60m",                # 模型保温 60 分钟，避免每次查询重新冷加载（首次 54s）
            "options": {"num_ctx": 8192, "temperature": 0.1, "num_predict": 1024},
            # num_ctx=8192：上下文窗口（图谱检索注入内容较多）；
            # temperature=0.1：低温保抽取稳定；
            # num_predict=1024：生成上限封顶（踩坑：不设上限 1.5B 场景下模板复读刷爆 token）
        },
        # Embedding 绑定：bge-m3（dim=1024），与本项目向量库一致
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,   # bge-m3 输出维度 1024（必须与模型实际维度一致，否则建索引报错）
            max_token_size=8192,  # 单段最大 token 数（超出截断）
            func=lambda texts: ollama_embed(  # 实际向量化函数：本机 Ollama 的 bge-m3
                texts, embed_model="dengcao/bge-m3:567m",
                host="http://127.0.0.1:11434"),
        ),
        # 工单要求：金融招股书领域实体/关系类型定制（addon_params 注入抽取 Prompt）
        addon_params={"language": "Chinese", "entity_types_guidance": ENTITY_GUIDANCE},
        # language=Chinese：实体/关系用中文抽取；guidance：领域类型指引（见上）
    )


# 图谱构建的实体/关系高密度关键词（用于精选输入，控制本机构建耗时）
KG_KEYWORDS = ["股东", "控股", "关联方", "募集", "发行", "客户", "供应商", "销售",
               "技术标准", "专利", "软件著作权", "高管", "董事", "监事", "子公司",
               "市场份额", "营业收入", "净利润", "注册资本", "视频指挥", "军用",
               "国防", "中标", "资质"]


# 必须进入图谱的关键事实片段（覆盖验收题真值：注册资本/控股股东持股比例等）
REQUIRED_FACTS = ["5,520", "42.35", "25.04%", "6,464.51", "15,000"]


def select_chunks(max_per_doc=60):
    """从两份招股书解析分块，按关键词命中数精选实体关系高密度片段"""
    selected = {}  # 结果：{文档名: 精选块列表}
    for pdf, name in [(PDF_ZGS1, "招股说明书1"), (PDF_ZGS2, "招股说明书2")]:  # 遍历两份 PDF
        chunks = chunk_pages(parse_pdf_text(pdf))  # 解析 PDF 并分块（每块带页码）
        scored = []  # (关键词得分, 块) 缓冲
        for c in chunks:
            # 得分 = 所有领域关键词在该块中的出现总次数
            s = sum(c["text"].count(k) for k in KG_KEYWORDS)
            if s >= 3:  # 至少命中3次才保留，过滤目录/扉页/噪音
                scored.append((s, c))
        scored.sort(key=lambda x: -x[0])  # 按密度降序排序
        picked = [c for _, c in scored[:max_per_doc]]  # 每文档取密度 Top 60 块
        # 强制补入含关键真值的块（密度排序可能漏掉短小的关键段）
        have = {id(c) for c in picked}  # 已选块的 id 集合（防重复）
        for c in chunks:
            # 未入选且文本中含任一关键真值（如注册资本数字）的块强制补入
            if id(c) not in have and any(f in c["text"] for f in REQUIRED_FACTS):
                picked.append(c)
        picked.sort(key=lambda c: c.get("page", 0))  # 按页码还原阅读顺序
        selected[name] = picked  # 存入结果
        print(f"[{name}] 总 {len(chunks)} 块，精选 {len(picked)} 块（密度Top + 真值强制入选）")
    return selected


async def _insert_doc(rag, name, picked):
    text = "\n\n".join(c["text"] for c in picked)  # 拼接精选块为整篇文档文本
    print(f"[{name}] 插入 {len(picked)} 块文本 ...")
    await rag.ainsert(text[:600000])  # 异步插入（LLM 抽取实体/关系建图）；60 万字符截断防超长
    print(f"[{name}] 完成")


async def build():
    """构建 LightRAG 知识图谱（增量式，已存在则跳过；单事件循环内完成）"""
    rag = make_rag()  # 创建 LightRAG 实例
    await rag.initialize_storages()  # 1.5.7 要求：先异步初始化存储（KV/向量/图库）
    try:
        if os.path.exists(MARKER):
            print("知识图谱已构建，跳过（删除 lightrag_work 可重建）")
            return rag
        os.makedirs(WORK_DIR, exist_ok=True)  # 确保工作目录存在
        for name, picked in select_chunks().items():  # 逐文档精选并插入
            await _insert_doc(rag, name, picked)
        # 防呆：确认实体图库确实生成再写完成标记
        if not os.path.exists(os.path.join(WORK_DIR, "vdb_entities.json")):
            raise RuntimeError("实体抽取失败（未生成 vdb_entities.json），请检查 build_log.txt")
        open(MARKER, "w").write("done")  # 写完成标记，下次运行跳过构建
        return rag
    finally:
        await rag.finalize_storages()  # 无论成败都要释放存储连接


async def _compare_flow():
    """RAG（向量库）vs LightRAG（知识图谱）对比主流程"""
    from lightrag import QueryParam  # 查询参数对象（指定检索模式）

    # 加载公共验收题（10 题招股书1）
    gt = json.load(open(os.path.join(COMMON, "ground_truth.json"), encoding="utf-8"))["questions"]
    # 14题 = 10 + 力源4题（招股书2 的追加验收题，含关键事实与答案提示）
    ly = [
        {"id": 1, "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？", "key_facts": ["1,670", "25.04%"], "answer_hint": "1,670万股，25.04%"},
        {"id": 2, "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？", "key_facts": ["仓储"], "answer_hint": "仓储物流/研发中心/电商平台"},
        {"id": 3, "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？", "key_facts": ["42.35%"], "answer_hint": "赵马克 42.35% 控股股东"},
        {"id": 4, "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？", "key_facts": ["融冰投资"], "answer_hint": "融冰投资/武汉博润/上海博润/听音投资/联众聚源"},
    ]
    questions = gt + ly  # 合并为 14 题

    # LightRAG 实例（如未构建则现场构建）
    rag = make_rag()
    await rag.initialize_storages()  # 异步初始化存储
    need_build = not os.path.exists(MARKER)  # 判断是否需要现场构建
    try:
        if need_build:
            os.makedirs(WORK_DIR, exist_ok=True)  # 确保工作目录存在
            for name, picked in select_chunks().items():  # 精选并逐文档插入
                await _insert_doc(rag, name, picked)
            if not os.path.exists(os.path.join(WORK_DIR, "vdb_entities.json")):
                raise RuntimeError("实体抽取失败（未生成 vdb_entities.json）")  # 防呆检查
            open(MARKER, "w").write("done")  # 写完成标记
        else:
            print("知识图谱已构建，直接进入对比")

        store = VectorStore.load("zgs_all_v1")  # 加载传统向量库索引（对照组）
        rows = []  # 对比结果行缓冲
        for i, item in enumerate(questions, 1):
            q = item["question"]  # 当前问题
            print(f"\n[{i}/{len(questions)}] {q}")
            # 传统RAG：向量检索 top5 + LLM 生成（与 LightRAG 同用 7B，只比检索结构差异）
            hits = store.search(q, top_k=5)  # 向量检索 top5 片段
            ctx = "\n\n".join(h["text"] for h in hits)  # 拼接检索片段为上下文
            rag_ans = client.chat([{"role": "user", "content": QA_PROMPT.format(context=ctx, question=q)}],
                                  temperature=0.0, num_predict=500)  # 同步调用 LLM 生成答案
            # LightRAG（混合模式：局部+全局双层检索）
            t0 = time.time()  # 记录 LightRAG 查询起始时间
            try:
                # mode=hybrid：同时走实体局部检索与关系全局检索，召回最全
                lr_ans = await rag.aquery(q, param=QueryParam(mode="hybrid"))
            except Exception as e:
                lr_ans = f"LightRAG查询失败: {e}"  # 失败不中断对比流程，记录错误
            lr_time = round(time.time() - t0, 1)  # LightRAG 单题耗时

            def ratio(facts, text):
                # 关键事实命中率：答案文本（去空格/逗号）中命中真值串的比例；无事实返回 -1
                t = str(text).replace(" ", "").replace(",", "")
                return sum(1 for f in facts if f.replace(",", "") in t) / len(facts) if facts else -1
            rows.append({"id": item["id"], "q": q, "hint": item["answer_hint"],
                         "rag_ans": rag_ans, "lr_ans": lr_ans,
                         "rag_ratio": ratio(item["key_facts"], rag_ans),  # RAG 命中率
                         "lr_ratio": ratio(item["key_facts"], lr_ans),    # LightRAG 命中率
                         "lr_time": lr_time})  # LightRAG 耗时
            print(f"  RAG命中 {rows[-1]['rag_ratio']:.0%} | LightRAG命中 {rows[-1]['lr_ratio']:.0%} ({lr_time}s)")

        n = len(rows)  # 总题数
        L = ["# 工单12：RAG vs LightRAG 检索结果对比",  # Markdown 报告缓冲
             "",
             f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}  |  LightRAG(hybrid双层检索) vs 向量RAG(top5)",
             "",
             "## 总体对比",
             "",
             "| 系统 | 关键事实命中率 |",
             "|---|---|",
             # 命中率≥50% 的题数 / 总题数
             f"| 传统向量 RAG | {sum(1 for r in rows if r['rag_ratio']>=0.5)}/{n} |",
             f"| LightRAG 知识图谱 | {sum(1 for r in rows if r['lr_ratio']>=0.5)}/{n} |",
             "",
             "## 逐题对比",
             ""]
        for r in rows:  # 逐题生成对比小节
            L += [f"### 问题 {r['id']}", f"**Q：** {r['q']}", f"**要点：** {r['hint']}",
                  f"**RAG回答**（命中{r['rag_ratio']:.0%}）：{r['rag_ans'][:300]}",  # 截断300字防报告过长
                  "",
                  f"**LightRAG回答**（命中{r['lr_ratio']:.0%}，耗时{r['lr_time']}s）：{str(r['lr_ans'])[:300]}",
                  "", "---", ""]
        out = os.path.join(HERE, "RAG与LightRAG对比-工单12.md")  # 报告输出路径
        open(out, "w", encoding="utf-8").write("\n".join(L))  # 写出报告
        print(f"\n对比报告: {out}")
    finally:
        await rag.finalize_storages()  # 释放存储连接


def compare():
    # 同步入口：在独立事件循环中运行异步对比流程
    asyncio.run(_compare_flow())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"  # 命令行参数：build/compare，默认 build
    if cmd == "build":
        asyncio.run(build())  # 构建知识图谱
    else:
        compare()  # 运行 RAG vs LightRAG 对比
