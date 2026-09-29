# -*- coding: utf-8 -*-
"""五角色知识库批量导入：老师/医生/律师/心理医生/科学家。

数据流（每个角色独立处理）：
  原始数据集 → 解析为统一 RoleRecord → ① SQL knowledge_chunks（原文/幂等）
                                      → ② BM25 增量索引（extra_docs.pkl 持久化）
                                      → ③ Milvus 向量（BGE-m3 编码，按角色打标）

用法（需在能连通 Milvus 的环境运行，如 WSL）：
  python ingest_roles.py                       # 五角色各导 2000 条，三写
  python ingest_roles.py --limit 500           # 每角色 500 条快速验证
  python ingest_roles.py --roles doctor,scientist
  python ingest_roles.py --skip-milvus         # 只写 SQL + BM25，不走向量
"""
from __future__ import annotations  # 启用 PEP 563 延迟注解求值，使 RoleRecord | None 写法在低版本 Python 也可用

import argparse  # 命令行参数解析（--limit / --roles / --skip-milvus）
import csv  # 解析医生数据集 CSV（question.csv / answer.csv）
import hashlib  # 生成 chunk_uid 内容哈希，作为三写共享幂等键
import json  # 解析律师 / 心理医生 JSON 数据集
import pickle  # 加载 extra_docs.pkl 持久化的 BM25 动态文档
import re  # _clean 中折叠连续空白字符
import sys  # main() 中校验失败时 sys.exit(1)
import traceback  # run() 中捕获解析异常后打印完整堆栈
from dataclasses import dataclass  # 用 @dataclass 简洁定义 RoleRecord
from pathlib import Path  # 跨平台路径拼接

from config import CACHE_DIR  # extra_docs.pkl 所在缓存目录
from database import KnowledgeChunk, SessionLocal, init_db  # ORM 表模型、会话工厂、建表入口
from logger import log  # 统一日志器（替代 print，便于定位问题）
from rag import SentencePair, TranslationRetriever, tokenize  # BM25 检索器、句对结构、中文分词

# 数据集根目录（data/uploads/<角色>/...）
UPLOADS = Path(__file__).resolve().parent / "data" / "uploads"  # 以脚本所在目录为基准拼出数据集根目录

# 长度上限：Milvus VARCHAR 最大 8192，保守截断；answer 单列控制在 1500
MAX_TEXT_LEN = 8000  # content 字段上限（BM25 语料 + Milvus text 共用，须小于 Milvus 8192）
MAX_ANSWER_LEN = 1500  # answer 列单独上限，控制单字段体积
MAX_LAWYER_CTX = 1800  # 律师裁判文书段落截断长度，过长会稀释检索相关性

# embedding 分批大小（CPU 内存与速度的平衡）
EMBED_BATCH = 32  # BGE-m3 每批编码 32 条，兼顾内存占用与吞吐


@dataclass  # 自动生成 __init__/__repr__，作为五角色统一的中间数据结构
class RoleRecord:
    """一条待入库的角色知识记录（五个 parser 的统一输出结构）。"""

    role_code: str  # 角色代码（teacher/doctor/lawyer/psychologist/scientist），用于分区与过滤
    source: str       # 来源文件名（落 SQL source 列）  # 记录数据出处，便于回溯与审计
    question: str     # 问题侧  # 作为 BM25 summary 与检索问题侧
    answer: str       # 答案侧  # 与 question 配对，作为内容答案
    content: str      # 拼接全文（BM25 语料 + Milvus text 必须完全一致，两路才能融合去重）  # 拼接后全文，作为向量与稀疏索引语料
    chunk_uid: str    # 内容哈希幂等键  # 由角色+全文 MD5 得到，三写去重的统一键


def _clean(text: str) -> str:
    """清洗：去首尾空白、折叠连续空白/换行。"""
    text = (text or "").strip()  # 容错空值并去首尾空白，避免空对象异常
    return re.sub(r"\s+", " ", text)  # 把连续空白（含换行/制表）折叠为单空格，统一 BM25/向量输入


def _make_record(role_code: str, source: str, question: str, answer: str,
                 qa_template: str = "问：{q}\n答：{a}") -> RoleRecord | None:
    """构造记录；内容过短或为空返回 None（过滤脏数据）。"""
    question = _clean(question)  # 清洗问题侧，统一空白
    answer = _clean(answer)[:MAX_ANSWER_LEN]  # 清洗答案侧并截断至 1500 字，避免单字段过长
    if len(question) + len(answer) < 8:  # 过滤近乎空的行  # 问+答合计不足 8 字视为脏数据
        return None
    content = qa_template.format(q=question, a=answer)[:MAX_TEXT_LEN]  # 按模板拼接全文并截断到 8000，保证 BM25 与 Milvus text 一致
    # 幂等键：角色 + 全文内容哈希，文件行序变化或重跑都不会产生重复
    uid = hashlib.md5(f"{role_code}|{content}".encode("utf-8")).hexdigest()  # 角色+内容做 MD5，作为三写共享幂等键
    return RoleRecord(role_code, source, question, answer, content, uid)  # 装配并返回统一记录


# =====================================================================
# 五个数据集解析器：各自负责一种私有格式 → 统一 RoleRecord
# =====================================================================

def parse_teacher(limit: int) -> list[RoleRecord]:
    """老师：中英翻译 TSV（四列 en_id, english, zh_id, chinese）。"""
    path = UPLOADS / "老师" / "中英翻译.tsv"  # 定位老师数据集文件
    source = "teacher:中英翻译.tsv"  # 落库 source 列的来源标识
    records: list[RoleRecord] = []  # 收集解析出的记录
    # utf-8-sig 自动去掉文件头 BOM
    with path.open(encoding="utf-8-sig") as f:  # 以 utf-8-sig 打开，自动剥离 BOM 避免首字段脏字符
        for line in f:  # 流式逐行读取，避免大文件整体载入
            parts = line.rstrip("\n").split("\t")  # 去行尾换行后按制表符切分四列
            if len(parts) != 4:  # 列数不是 4 视为脏行
                continue
            _, english, _, chinese = parts  # 仅取英文与中文两列，id 列丢弃
            rec = _make_record(  # 用英文作 question、中文作 answer 构造记录
                "teacher", source, english, chinese,
                qa_template="English: {q}\nChinese: {a}",  # 自定义拼接模板，体现中英对照语义
            )
            if rec:  # _make_record 返回 None 表示脏数据已过滤
                records.append(rec)  # 加入结果集
            if len(records) >= limit:  # 达到上限提前结束
                break
    return records  # 返回该角色的全部记录


def parse_doctor(limit: int) -> list[RoleRecord]:
    """医生：question.csv + answer.csv 按 question_id 关联，取每个问题的第一条回答。"""
    q_file = UPLOADS / "医生" / "question.csv"  # 问题表路径
    a_file = UPLOADS / "医生" / "answer.csv"  # 答案表路径
    source = "doctor:question.csv+answer.csv"  # 来源标识两文件合并

    questions: dict[str, str] = {}  # question_id → 问题内容（保序）  # 用 dict 保序记录问题，键为 id
    with q_file.open(encoding="utf-8", newline="") as f:  # newline="" 让 csv 模块正确处理换行
        for row in csv.DictReader(f):  # 按表头读字典行
            qid = (row.get("question_id") or "").strip()  # 取问题 id 并清洗
            content = (row.get("content") or "").strip()  # 取问题内容并清洗
            if qid and content:  # id 与内容都有效才收录
                questions[qid] = content

    first_answer: dict[str, str] = {}  # question_id → 首条回答  # 只保留每个问题的第一条答案
    # answer.csv 68MB，流式读取避免整体载入内存
    with a_file.open(encoding="utf-8", newline="") as f:  # 大文件流式读，控制内存
        for row in csv.DictReader(f):
            qid = (row.get("question_id") or "").strip()  # 取该答案对应的问题 id
            if qid in questions and qid not in first_answer:  # 只保留第一条  # 仅当问题存在且尚未收录过答案
                content = (row.get("content") or "").strip()  # 取答案内容
                if content:  # 空答案跳过
                    first_answer[qid] = content

    records: list[RoleRecord] = []
    for qid, question in questions.items():  # 按问题文件顺序取  # 保序遍历，保持原始顺序
        answer = first_answer.get(qid)  # 取该问题的首条答案
        if not answer:
            continue  # 无答案的问题跳过  # 没有答案匹配的问题直接跳过
        rec = _make_record("doctor", source, question, answer)  # 构造医生记录
        if rec:
            records.append(rec)
        if len(records) >= limit:  # 达到上限提前结束
            break
    return records


def parse_lawyer(limit: int) -> list[RoleRecord]:
    """律师：SQuAD 风格 JSON（data[].paragraphs[].context），裁判文书段落入库。"""
    path = UPLOADS / "律师" / "big_train_data.json"  # 律师数据集路径
    source = "lawyer:big_train_data.json"  # 来源标识
    with path.open(encoding="utf-8") as f:  # 整体载入 JSON（文件不大）
        payload = json.load(f)  # 解析为 dict

    records: list[RoleRecord] = []
    for item in payload.get("data", []):  # 遍历 SQuAD 顶层 data 数组
        for para in item.get("paragraphs", []):  # 遍历每条数据的 paragraphs
            context = _clean(para.get("context", ""))  # 取裁判文书段落并清洗
            if len(context) < 20:  # 段落过短视为噪声跳过
                continue
            context = context[:MAX_LAWYER_CTX]  # 截断到 1800 字，控制单条长度
            # casename 在 paragraph 层级；问题侧用案由+段落开头，答案侧为整段裁判文书
            casename = _clean(para.get("casename", ""))  # 取案由（如有）
            title = casename or context[:30]  # 案由缺失则用段落前 30 字作题
            rec = _make_record("lawyer", source, f"【{title}】{context[:60]}", context)  # 问题侧放标题+开头，答案侧放整段
            if rec:
                records.append(rec)
            if len(records) >= limit:  # 达到上限直接返回，避免继续遍历
                return records
    return records


def parse_psychologist(limit: int) -> list[RoleRecord]:
    """心理医生：CPsyCounD.json（list，instruction/input/output/history）。"""
    path = UPLOADS / "心理医生" / "CPsyCounD.json"  # 心理数据集路径
    source = "psychologist:CPsyCounD.json"  # 来源标识
    with path.open(encoding="utf-8") as f:
        data = json.load(f)  # 整体载入为 list

    records: list[RoleRecord] = []
    for item in data:  # 逐条样本遍历
        question = _clean(item.get("instruction", ""))  # 取 instruction 作为问题主干
        extra = _clean(item.get("input", ""))  # 取 input 作为补充上下文
        if extra:  # 部分样本 input 为空  # 有补充则拼到问题后
            question = f"{question} {extra}".strip()
        answer = item.get("output", "")  # output 作为答案
        rec = _make_record("psychologist", source, question, answer)  # 构造心理记录
        if rec:
            records.append(rec)
        if len(records) >= limit:
            break
    return records


def parse_scientist(limit: int) -> list[RoleRecord]:
    """科学家：HuggingFace arrow（train 两个分片），pyarrow 流式分批读取。"""
    import pyarrow.ipc as ipc  # 延迟导入 pyarrow，避免未装该库时整脚本无法加载

    arrow_dir = (  # arrow 文件所在目录
        UPLOADS
        / "科学家" / "TheMrguiller___science_qa" / "default" / "0.0.0"
        / "180af444885435eb3b890bad87348369cd144800"
    )
    shards = [  # train 集的两个分片
        arrow_dir / "science_qa-train-00000-of-00002.arrow",
        arrow_dir / "science_qa-train-00001-of-00002.arrow",
    ]
    source = "scientist:science_qa-train.arrow"  # 来源标识
    records: list[RoleRecord] = []

    for shard in shards:  # 逐分片处理
        if len(records) >= limit:  # 已达上限就跳出分片循环
            break
        with ipc.open_stream(shard) as reader:  # streaming reader 逐批读取，省内存  # 流式读取 arrow，避免整表载入
            for batch in reader:  # 按 RecordBatch 迭代
                rows = batch.to_pylist()  # 把批次转成 Python dict 列表
                for row in rows:  # 逐行处理
                    question = _clean((row.get("question") or "").replace("[QUESTION]", ""))  # 去掉占位符 [QUESTION]
                    choices = _clean((row.get("choices") or "").replace("[OPTIONS]", ""))  # 去掉占位符 [OPTIONS]
                    answer = _clean(row.get("answer") or "")  # 取标准答案
                    solution = _clean(row.get("solution") or "")  # 取解题过程
                    full_answer = f"{answer}。{solution}".strip("。")  # 答案+解法拼接，去多余句号
                    full_question = f"{question} 选项：{choices}" if choices else question  # 有选项则拼到问题后
                    rec = _make_record("scientist", source, full_question, full_answer)  # 构造科学家记录
                    if rec:
                        records.append(rec)
                    if len(records) >= limit:  # 达到上限直接返回
                        return records
    return records


# 角色 → 解析器注册表
PARSERS = {  # 角色 code 到解析函数的映射，run() 据此分派
    "teacher": parse_teacher,
    "doctor": parse_doctor,
    "lawyer": parse_lawyer,
    "psychologist": parse_psychologist,
    "scientist": parse_scientist,
}


# =====================================================================
# BM25 增量索引（直接基于 extra_docs.pkl，避免依赖主 TSV 的跨平台路径）
# =====================================================================

def load_extra_retriever() -> TranslationRetriever:
    """加载只含动态文档的检索器（extra_docs.pkl）；不存在则空检索器。"""
    from rank_bm25 import BM25Okapi  # 延迟导入 BM25 实现，避免未用时加载

    extra_file = CACHE_DIR / "extra_docs.pkl"  # 持久化动态文档的 pkl 路径
    pairs: list[SentencePair] = []  # 已有句对集合
    if extra_file.exists():  # 历史持久化存在则载入
        with extra_file.open("rb") as f:
            pairs = pickle.load(f)  # 反序列化已有句对
    if pairs:  # 已有语料则基于其分词构造 BM25 索引
        bm25 = BM25Okapi([tokenize(p.as_text()) for p in pairs])  # 用每条句对的正文分词后构造 BM25
    else:
        bm25 = None  # add_pairs 内部 _rebuild 时会用新语料重新构造  # 留空，后续 add_pairs 时重建
    return TranslationRetriever(pairs, bm25)  # 返回装好语料与索引的检索器


# =====================================================================
# 三写主流程
# =====================================================================

def write_sql(role_code: str, records: list[RoleRecord]) -> list[RoleRecord]:
    """① 写 SQL knowledge_chunks；按 chunk_uid 幂等，返回真正新增的记录。

    幂等三重保障：
    - 查询已存在的 chunk_uid 过滤；
    - records 内部按 chunk_uid 去重（避免同批重复，如律师裁判文书相同段落）；
    - SQLite 用 INSERT OR IGNORE 兜底唯一约束冲突。
    """
    with SessionLocal() as db:  # 起一个 SQL 会话，作用域结束自动关闭
        existed = {  # 一次性查出该角色已存在的所有 chunk_uid，避免逐条查询
            uid for (uid,) in db.query(KnowledgeChunk.chunk_uid)
            .filter(KnowledgeChunk.role_code == role_code).all()
        }
        # 按 chunk_uid 去重（取首次出现），并跳过数据库已存在的
        seen: set[str] = set()  # 本批内已见 uid 集合，防止同批重复
        new_records: list[RoleRecord] = []  # 真正需要写入的新增记录
        for r in records:  # 遍历传入记录做去重
            if r.chunk_uid in existed or r.chunk_uid in seen:  # 已在库或本批已见则跳过
                continue
            seen.add(r.chunk_uid)  # 标记本批已见
            new_records.append(r)  # 加入待写列表

        from sqlalchemy.dialects.sqlite import insert as sqlite_insert  # 延迟导入 SQLite 专用 insert 以支持 OR IGNORE

        for i in range(0, len(new_records), 500):  # 按 500 条/批提交，避免单事务过大
            batch = new_records[i : i + 500]  # 切出当前批次
            stmt = sqlite_insert(KnowledgeChunk).values([  # 构造批量插入语句
                {
                    "role_code": r.role_code,
                    "source": r.source,
                    "question": r.question,
                    "answer": r.answer,
                    "content": r.content,
                    "chunk_uid": r.chunk_uid,
                }
                for r in batch
            ])
            # ON CONFLICT DO NOTHING：唯一约束冲突时静默跳过
            stmt = stmt.prefix_with("OR IGNORE")  # 加 OR IGNORE 前缀，冲突时静默跳过兜底
            db.execute(stmt)  # 执行批量插入
            db.commit()  # 每批提交一次，保证已写数据落盘
    return new_records  # 返回真正新增的记录，供 BM25/Milvus 后续使用


def write_bm25(retriever: TranslationRetriever, records: list[RoleRecord]) -> int:
    """② 增量写 BM25；按 content 去重防脏数据，add_pairs 自动持久化 pkl。"""
    existing = {p.as_text() for p in retriever.pairs}  # 取已存在语料的正文集合，用于按 content 去重
    pairs = [  # 构造 SentencePair 列表
        SentencePair(
            en_id=f"{r.role_code}-{r.chunk_uid[:8]}",  # 拼出唯一 en_id，前缀角色便于排查
            english=f"[DOC] {r.question[:60]}",  # [DOC] 前缀：as_text 返回正文 content  # 加 [DOC] 前缀，使 as_text() 返回正文 content
            zh_id=r.chunk_uid[:8],  # 取 uid 前 8 位作为短 id
            chinese=r.content,                    # 与 Milvus text 完全一致  # 与 Milvus text 字段完全一致，保证两路融合去重
            source=r.source,  # 来源标签便于回溯
            summary=r.question[:80],  # 摘要字段截断到 80 字
        )
        for r in records
        if r.content not in existing  # 已存在 content 的跳过，避免重复入库
    ]
    if pairs:  # 有新增才调用 add_pairs，内部重建 BM25 并持久化 pkl
        retriever.add_pairs(pairs)  # 内部自动重建索引并写回 extra_docs.pkl
    return len(pairs)  # 返回新增条数，供汇总打印


def write_milvus(records: list[RoleRecord]) -> int:
    """③ BGE-m3 分批编码后写 Milvus（带 role_code/chunk_uid），按 chunk_uid 幂等。

    独立于 SQL 做幂等：先查询 Milvus 已存在的 chunk_uid，只编码并写入缺失的。
    这样即使 SQL 已存在而 Milvus 缺失（如之前向量化失败），也能补写。
    """
    from embeddings import embed_texts  # 延迟导入嵌入函数，避免脚本启动加载模型
    from vector_store import ensure_collection, upsert_texts  # 延迟导入 Milvus 集合与 upsert 工具

    try:
        from tqdm import tqdm  # 优先使用 tqdm 显示进度
    except ImportError:  # tqdm 缺失时退化为无进度条  # 没装 tqdm 则用 identity 函数降级
        def tqdm(it, **kwargs):
            return it

    col = ensure_collection()  # 确保 Milvus 集合存在并返回句柄
    if col is None:  # 集合不可用则跳过向量写入
        log.warning("Milvus 不可用，跳过向量写入")
        return 0

    # 查询该角色已存在的 chunk_uid，做幂等过滤。
    # 注意：Milvus query 的 offset+limit 总窗口上限 16384，普通分页翻不完
    # 大分区（昨晚日志里正是这个错）；改用 query_iterator 游标分批拉取。
    role_code = records[0].role_code  # 取第一条记录的角色，作为分区与过滤条件
    existing_uids: set[str] = set()  # 收集 Milvus 已存在 uid
    try:
        from vector_store import _list_partitions  # 延迟导入分区列举函数

        partition_names = (  # 角色分区存在则只查该分区，否则全量
            [role_code] if role_code in _list_partitions(col) else None
        )  # 分区不存在则全量写入  # 分区不存在时返回 None 表示不限定分区
        try:
            it = col.query_iterator(  # 优先使用游标迭代器，规避 16384 上限
                batch_size=1000, expr=f'role_code == "{role_code}"', partition_names=partition_names
            )
            while True:  # 不断 next 直到无数据
                res = it.next()  # 拉取一批
                if not res:  # 空批表示结束
                    it.close()  # 关闭迭代器释放服务端资源
                    break
                existing_uids.update(r["chunk_uid"] for r in res)  # 累加本批 uid
        except AttributeError:  # 旧版 pymilvus 无 query_iterator：单窗口兜底（最多 16384 条）  # 旧版 pymilvus 无迭代器 API 时退化
            res = col.query(  # 单次 query 兜底
                expr=f'role_code == "{role_code}"',
                output_fields=["chunk_uid"],  # 只取 chunk_uid 字段省流量
                partition_names=partition_names,
                limit=16384,  # 取满单窗口上限
            )
            existing_uids.update(r["chunk_uid"] for r in res)
    except Exception as exc:  # 查询失败则降级为全量写入
        log.warning("Milvus 查询已有 chunk_uid 失败，将全量写入: %s", exc)

    pending = [r for r in records if r.chunk_uid not in existing_uids]  # 只保留 Milvus 缺失的记录
    if not pending:  # 全部已存在则直接返回 0
        return 0

    total = 0  # 累计写入条数
    texts = [r.content for r in pending]  # 抽出待编码文本
    uids = [r.chunk_uid for r in pending]  # 抽出对应 uid，用于 upsert 标记
    source = pending[0].source  # 取首条 source 作为本批来源
    for i in tqdm(range(0, len(texts), EMBED_BATCH),  # 按 EMBED_BATCH 分批并显示进度
                  desc=f"{role_code} 向量化", unit="batch"):
        batch_texts = texts[i : i + EMBED_BATCH]  # 切出本批文本
        vectors = embed_texts(batch_texts)  # 模型单例，仅首批加载  # BGE-m3 编码本批文本
        if not vectors:  # 编码失败返回空则中止本次写入
            log.warning("embedding 返回空，跳过 Milvus 写入：%s", role_code)
            return 0
        total += upsert_texts(  # 累加 upsert 写入条数
            batch_texts, vectors,
            source=source,
            role_code=role_code,
            chunk_uids=uids[i : i + EMBED_BATCH],  # 同步切出本批 uid
        )
    return total  # 返回总写入条数


def run(limit: int, roles: list[str], skip_milvus: bool) -> None:
    """导入主入口：逐角色 解析 → SQL → BM25 → Milvus。"""
    init_db()  # 确保 knowledge_chunks 表已建  # 建表（幂等），避免首次运行缺表
    retriever = load_extra_retriever()  # 载入 BM25 动态文档检索器
    print(f"BM25 动态文档起点：{len(retriever.pairs)} 条\n")  # 打印起点条数，便于核对增量

    summary_rows = []  # 收集每个角色的统计行
    for role_code in roles:  # 逐角色处理
        parser = PARSERS[role_code]  # 取该角色的解析器
        print(f"===== [{role_code}] 解析数据集中 ... =====")
        try:
            records = parser(limit)  # 调用解析器，得到 RoleRecord 列表
        except Exception as exc:  # 解析失败不影响其他角色
            print(f"[{role_code}] 解析失败，跳过：{exc}")
            traceback.print_exc()  # 打印完整堆栈定位问题
            summary_rows.append((role_code, 0, 0, 0, 0))  # 全 0 占位
            continue
        print(f"[{role_code}] 解析得到 {len(records)} 条")

        new_records = write_sql(role_code, records)  # 写 SQL 并拿到真正新增记录
        print(f"[{role_code}] SQL 新增 {len(new_records)} 条（已存在跳过 {len(records) - len(new_records)} 条）")

        bm25_n = write_bm25(retriever, new_records)  # 用新增记录增量写 BM25
        print(f"[{role_code}] BM25 新增 {bm25_n} 条")

        milvus_n = 0  # 默认 0，失败或跳过时保持
        if not skip_milvus and records:  # 未跳过且有记录才尝试写 Milvus
            try:
                milvus_n = write_milvus(records)  # 传全部 records，内部按 chunk_uid 独立幂等  # 注意传全部 records 而非 new_records，独立做幂等以补缺失向量
            except Exception as exc:  # Milvus 失败不回滚 SQL/BM25
                print(f"[{role_code}] Milvus 写入失败（SQL/BM25 已成功，可稍后重跑补向量）：{exc}")
                traceback.print_exc()
        print(f"[{role_code}] Milvus 写入 {milvus_n} 条\n")
        summary_rows.append((role_code, len(records), len(new_records), bm25_n, milvus_n))  # 收集统计行

    print("=" * 70)  # 打印汇总标题分隔
    print("导入汇总")
    print(f"{'角色':<14}{'解析':>6}{'SQL新增':>10}{'BM25':>8}{'Milvus':>8}")  # 表头
    for role_code, parsed, sql_n, bm25_n, milvus_n in summary_rows:  # 逐行打印汇总
        print(f"{role_code:<14}{parsed:>6}{sql_n:>10}{bm25_n:>8}{milvus_n:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description="五角色知识库导入 SQL/BM25/Milvus")  # 构造命令行解析器
    parser.add_argument("--limit", type=int, default=2000, help="每个角色最多导入条数（默认 2000）")  # 每角色上限
    parser.add_argument(
        "--roles", type=str, default=",".join(PARSERS.keys()),  # 默认全部角色
        help="逗号分隔的角色列表，默认全部",
    )
    parser.add_argument("--skip-milvus", action="store_true", help="只写 SQL+BM25，跳过向量化")  # 开关参数
    args = parser.parse_args()  # 解析命令行

    roles = [r.strip() for r in args.roles.split(",") if r.strip()]  # 拆分并清洗角色列表
    bad = [r for r in roles if r not in PARSERS]  # 找出未知角色
    if bad:  # 有未知角色则提示并退出
        print(f"未知角色：{bad}；可选：{list(PARSERS.keys())}")
        sys.exit(1)  # 非零退出便于脚本编排捕获
    run(args.limit, roles, args.skip_milvus)  # 进入主流程


if __name__ == "__main__":  # 直接运行本脚本时进入 main
    main()


# =====================================================================
# 知识点说明（RAG：多角色知识库 / 索引构建 / 混合存储）
# ---------------------------------------------------------------------
# 1. 多角色知识库的隔离方式：Milvus 单集合 + 每角色一个分区
#    （partition），写入 insert(partition_name=role_code) 定向落分区，
#    检索 search(partition_names=[role_code]) 只扫该分区；SQL 侧
#    knowledge_chunks 用 role_code 索引列做关系查询。相比"每角色一个
#    集合"，单集合多分区运维简单、跨角色统计方便；相比纯标量过滤，
#    分区物理裁剪少扫无关向量。
# 2. 双存储分工：SQL（SQLite/MySQL）存结构化原文，支持事务、精确查询、
#    后台管理与审计；Milvus 只存向量 + 少量回表字段，专做 ANN 语义召回。
#    两者用 chunk_uid（内容 MD5）一一对应——这是工业界 RAG 的标准拆法：
#    "向量库负责找得准，关系库负责管得住"。
# 3. 幂等导入：chunk_uid 由"角色+全文内容"哈希得到，重跑/中断续跑时
#    SQL 先查已存在键跳过，BM25 按 content 去重，Milvus 只对 SQL 新增
#    记录编码写入——三路都不会产生重复数据。
# 4. RAG 离线索引流水线（本脚本）：原始数据（CSV/JSON/Arrow/TSV）→
#    解析清洗 → 统一文本块 → BM25 稀疏索引 + BGE-m3 稠密向量双写，
#    与在线检索（rag.hybrid_search：稀疏+稠密融合 → rerank 精排）对应。
# 5. 分批工程实践：embedding 按 32 条/批编码（控内存、显进度）；SQL 按
#    500 条/批提交（避事务过大）；大文件（68MB answer.csv、arrow）一律
#    流式读取，不整体载入内存。
# =====================================================================
