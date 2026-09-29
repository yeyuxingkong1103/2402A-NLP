"""统一评测入口：retrieval 测检索，ragas 测生成质量。

常用命令：
  python -m app.evaluate retrieval --compare-rerank
  python -m app.evaluate ragas
  python -m app.evaluate ragas --run --compare-rerank

retrieval 不生成答案、不调用付费 API；ragas 默认也只校验，只有 --run 才付费。
"""
# 本文件同时提供两套命令行流程：
# 1. retrieval：只检查“检索到了哪些知识块”，不会让大模型生成答案；
# 2. ragas：检查“生成答案的质量”，默认只校验输入，必须显式传入 --run 才会调用模型。
# 对初学者来说，可以从文件末尾的 main() 开始顺着调用关系阅读。

# postponed evaluation of annotations：推迟解析类型注解，避免注解在导入阶段立即求值。
from __future__ import annotations

# argparse 用来声明、解析命令行参数，例如 --top-k 和 --run。
import argparse
# asyncio 用来并发执行一行样本中的多个异步 RAGAS 指标。
import asyncio
# copy.deepcopy 创建提示词对象的独立副本，避免修改 RAGAS 内置的共享提示词。
import copy
# hashlib 用 SHA-256 记录评测集/核心代码版本，便于复现实验。
import hashlib
# importlib.metadata 查询当前 Python 环境中已安装包的版本号。
import importlib.metadata
# json 负责读取评测集、写 JSON 报告，以及在 Markdown 中嵌入配置。
import json
# math.isfinite 用来排除 NaN 和正负无穷等无效分数。
import math
# os 负责读取/临时设置环境变量，例如离线模式和 API 配置。
import os
# sys 提供命令行参数、标准错误输出以及 Python 模块搜索路径。
import sys
# time.perf_counter 提供适合统计耗时的高精度单调时钟。
import time
# datetime.now(timezone.utc) 为报告生成带 UTC 时区的时间戳。
from datetime import datetime, timezone
# Path 用面向对象的方式拼接、读取和写入文件路径。
from pathlib import Path
# urlsplit 只拆解评测服务 URL，写报告时不记录用户名、密码或查询参数。
from urllib.parse import urlsplit

# 当前文件位于 app/internal；parents[2] 才是项目根目录。
BASE = Path(__file__).resolve().parents[2]
# 检索评测统一使用的四个指标名，元组保证名称和顺序固定。
RETRIEVAL_METRICS = ("hit_at_k", "recall_at_k", "precision_at_k", "mrr")
# 生成质量评测统一使用的四个 RAGAS 指标名。
RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
# 项目只接受这个 RAGAS 版本，避免不同版本的 API 或评分方式造成结果不可比。
EXPECTED_RAGAS = "0.2.15"


class SetupError(RuntimeError):
    """表示可预期的评测初始化失败。

    这类异常的文字由本项目自己编写，可以安全地展示给用户；其他未知异常只展示
    异常类型，避免把 API Key、服务响应或本地路径等敏感细节写进报告。
    """


# ---------- 检索评测：不生成答案，不调用在线模型 ----------
def retrieval_score_case(retrieved, relevant, top_k=4):
    """计算一道检索题的 Hit、Recall、Precision 和 MRR。

    参数：
        retrieved: 检索器按相关性从高到低返回的知识块编号序列。
        relevant: 人工标注为相关的知识块编号序列；空序列表示越界问题。
        top_k: 只统计前 K 个结果，默认取前 4 个。
    返回：包含四项检索指标、越界题结果和命中块编号的字典。
    """
    # K 至少为 1，否则“前 K 个结果”和 Precision@K 都没有意义。
    if top_k < 1:
        raise ValueError("top_k必须为正整数")
    # dict.fromkeys 按首次出现顺序去重，再转回列表并截取前 K 项。
    # 去重可避免同一知识块重复出现时虚增命中数。
    ranked = list(dict.fromkeys(retrieved))[:top_k]
    # 集合适合做交集和成员判断，也会消除人工标注中的重复编号。
    expected = set(relevant)
    # 没有相关块表示问题超出知识库范围，此时四个普通检索指标不适用。
    if not expected:
        # ranked 为空说明系统正确地“什么都没检索到”；bool 转 float 得到 1.0/0.0。
        return {**{name: None for name in RETRIEVAL_METRICS},
                "out_of_scope_correct": float(not ranked), "matched": []}
    # 预测结果与人工标注取交集，得到真正命中的知识块集合。
    matched = set(ranked) & expected
    # enumerate(..., 1) 让名次从 1 开始；next 找第一个命中项，找不到则返回 None。
    first_rank = next((index for index, item in enumerate(ranked, 1) if item in expected), None)
    # Hit 只关心是否至少命中一个；Recall 看相关块找回比例；Precision 的分母固定为 K；
    # MRR 是首个命中名次的倒数。matched 排序后写报告，保证输出稳定。
    return {"hit_at_k": float(bool(matched)), "recall_at_k": len(matched) / len(expected),
            "precision_at_k": len(matched) / top_k, "mrr": 1 / first_rank if first_rank else 0.0,
            "out_of_scope_correct": None, "matched": sorted(matched)}


def retrieval_summarize(results):
    """按检索配置汇总逐题结果。

    参数 results 是 run_retrieval() 产生的逐题字典列表；返回值以 variant
    （hybrid 或 hybrid_rerank）为键，包含成功/失败数、指标均值和平均耗时。
    """
    # summary 最终会直接写入 JSON 报告。
    summary = {}
    # 集合去重得到所有配置名，sorted 保证报告顺序稳定。
    for variant in sorted({row["variant"] for row in results}):
        # 先筛出当前配置的全部记录，再筛出真正完成的记录。
        rows = [row for row in results if row["variant"] == variant]
        completed = [row for row in rows if row["status"] == "ok"]
        # 失败数用总数减成功数，失败记录不会作为 0 分混进平均值。
        item = {"total": len(rows), "completed": len(completed), "failed": len(rows) - len(completed)}
        # 普通检索指标和越界正确率采用相同的“仅对适用项求均值”规则。
        for name in (*RETRIEVAL_METRICS, "out_of_scope_correct"):
            # None 表示该指标不适用于此题，因此过滤掉；数值 0 必须保留。
            values = [row["scores"][name] for row in completed if row["scores"][name] is not None]
            item[name] = sum(values) / len(values) if values else None
            # 同时记录分母，让读者知道均值实际基于多少题。
            item[name + "_count"] = len(values)
        # elapsed_ms 已经是毫秒；只有成功题参与延迟均值。
        item["mean_retrieval_ms"] = (sum(row["elapsed_ms"] for row in completed) / len(completed)
                                      if completed else None)
        summary[variant] = item
    return summary


def load_retrieval_cases(path):
    """读取并校验检索评测集，返回题目字典列表。

    path 可以是字符串或 Path。文件必须是非空 JSON 数组，每题需要唯一 id、
    非空 question，以及由非负整数构成的 relevant_chunk_indexes。
    """
    # utf-8-sig 同时兼容普通 UTF-8 和带 BOM 的 Windows UTF-8 文件。
    cases = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    # 顶层不是列表或列表为空时，后续评测没有明确语义，立即报错。
    if not isinstance(cases, list) or not cases:
        raise ValueError("评测集必须是非空列表")
    # seen 保存已出现的题目 id，用来发现重复项。
    seen = set()
    for case in cases:
        # 每道题必须是对象，且 id 必须是非空字符串。
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"]:
            raise ValueError("每题必须有非空id")
        if case["id"] in seen:
            raise ValueError("题目id重复")
        seen.add(case["id"])
        # strip 后仍为空可拦截只包含空格、换行符的无效问题。
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError("题目问题不能为空")
        # get 在字段缺失时返回 None，随后也会被判为不合法。
        indexes = case.get("relevant_chunk_indexes")
        # type(index) is int 特意不接受 bool；Python 中 bool 虽是 int 子类，但不是合法块号。
        if not isinstance(indexes, list) or any(type(index) is not int or index < 0 for index in indexes):
            raise ValueError("relevant_chunk_indexes必须是非负整数列表")
    return cases


def retrieval_markdown(report):
    """把检索报告字典渲染为便于阅读的 Markdown 字符串。"""
    # 先写标题、状态和评测口径；列表中的每个元素最终对应一行。
    lines = ["# 检索指标评测报告（不是RAGAS）", "", f"运行状态：{report['status']}",
             f"时间：{report['created_at']}；TopK={report['top_k']}；题目数={report['case_count']}",
             "", "本报告只检验检索，不生成答案、不调用裁判。越界题单列统计；失败不计作0分。",
             "精排对照必须真实加载权重；平均延迟受首次模型热身影响，不代表压力测试QPS。", ""]
    # 只有失败报告才通常含 error 字段；get 可避免字段不存在时抛 KeyError。
    if report.get("error"):
        lines.extend(["错误：" + report["error"], ""])
    # 追加 Markdown 表头和分隔行。
    lines.extend(["| 模式 | 成功/总数 | Hit@K | Recall@K | Precision@K | MRR | 越界空检索正确率 |",
                  "|---|---:|---:|---:|---:|---:|---:|"])

    def display(value):
        """把可选浮点数格式化为四位小数；None 显示为长横线。"""
        return "—" if value is None else f"{value:.4f}"

    # get(..., {}) 兼容初始化阶段尚无 summary 的报告。
    for variant, item in report.get("summary", {}).items():
        values = [display(item[name]) for name in (*RETRIEVAL_METRICS, "out_of_scope_correct")]
        lines.append(f"| {variant} | {item['completed']}/{item['total']} | " + " | ".join(values) + " |")
    # 接着添加逐题表，便于定位具体失败或命中情况。
    lines.extend(["", "## 逐题记录", "", "| 题目ID | 模式 | 状态 | 命中块 | 耗时ms |",
                  "|---|---|---|---|---:|"])
    for row in report["results"]:
        # 失败记录可能没有 retrieved_chunk_indexes，所以用 get 提供空列表。
        lines.append(f"| {row['id']} | {row['variant']} | {row['status']} | "
                     f"{row.get('retrieved_chunk_indexes', [])} | {row['elapsed_ms']:.1f} |")
    # 最后写入解释指标口径的固定备注。
    lines.extend(["", "参考答案和块号来自人工标注，语料重新分块后必须重新核对。",
                  "Precision@K的分母固定为K；空标注题仅用于判断检索是否应为空，不表示医生回答拒答质量。"])
    # join 组合所有行，并在文件末尾保留换行符。
    return "\n".join(lines) + "\n"


def save_retrieval_report(report, output, summarize_function=None):
    """同时保存 JSON 与 Markdown 两种检索报告。

    report 会原地补上 summary；output 是不带扩展名的路径前缀。
    函数无返回值，写入失败时让文件异常继续向上传播。
    """
    # 入口可传入教学版汇总函数；未传时仍使用本模块的默认实现。
    summarize = summarize_function or retrieval_summarize
    report["summary"] = summarize(report["results"])
    # parents=True 会连同缺失的父目录一起创建；已存在时不报错。
    output.parent.mkdir(parents=True, exist_ok=True)
    # JSON 供程序读取，Markdown 供人查看；allow_nan=False 防止写出非标准 JSON。
    contents = ((".json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)),
                (".md", retrieval_markdown(report)))
    for suffix, body in contents:
        # with_suffix 把路径前缀替换为目标后缀，例如 report -> report.json。
        target = output.with_suffix(suffix)
        # 先写 .tmp，再用 replace 原子替换目标，可降低中途退出留下半份报告的风险。
        temporary = target.with_suffix(suffix + ".tmp")
        temporary.write_text(body, encoding="utf-8")
        temporary.replace(target)


def report_prefix(output, state):
    """返回带状态后缀的新路径前缀。

    例如 output=report、state=failed 时返回 report_failed；这样运行中或失败报告
    不会覆盖最近一次成功报告。
    """
    return output.with_name(output.name + "_" + state)


def build_retrieval_engine(project_root, use_rerank):
    """构建只用于检索的 RAG 引擎。

    project_root 是项目根目录；use_rerank 决定是否启用重排序模型。
    返回 app.single_app.RAG 实例。函数只创建检索能力，不在这里生成答案。
    """
    # 把目标项目根目录放到导入路径首位，确保下面导入的是该项目的 app。
    sys.path.insert(0, str(project_root))
    # setdefault 仅在调用者未设置时启用离线模式，禁止 Hugging Face 自动联网下载。
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # 延迟导入：只有真正运行检索时才加载业务模块及其较重依赖。
    from app.single_app import RAG
    # 保存两个可能被临时修改的环境变量，finally 中会完整恢复。
    previous = {name: os.environ.get(name) for name in ("LLM_API_KEY", "RAG_RERANK_ENABLED")}
    # RAG 初始化要求存在 Key；检索流程不会调用 LLM，因此用明确的占位值即可。
    if not os.environ.get("LLM_API_KEY"):
        os.environ["LLM_API_KEY"] = "retrieval-only-no-api-call"
    # 业务代码用字符串环境变量控制是否加载重排序器。
    os.environ["RAG_RERANK_ENABLED"] = "true" if use_rerank else "false"
    try:
        # with_memory=False 表示评测不读取或写入 Redis 对话记忆。
        return RAG(with_memory=False)
    finally:
        # 无论初始化成功还是异常，都把进程环境恢复到调用前状态。
        for name, value in previous.items():
            if value is None:
                # 原来不存在的变量直接删除。
                os.environ.pop(name, None)
            else:
                # 原来存在的变量恢复原值。
                os.environ[name] = value


def run_retrieval(args, score_function=None, summarize_function=None,
                  engine_builder=None):
    """执行完整的检索指标评测，成功返回 0，失败返回非零退出码。

    args 是 parse_retrieval_args() 返回的 Namespace，包含评测集、项目路径、
    输出路径、TopK 以及是否启用/对比重排序等设置。
    """
    # 教学入口可显式传入自己的指标函数；普通调用继续使用这里的默认实现。
    score_case = score_function or retrieval_score_case
    summarize = summarize_function or retrieval_summarize
    make_engine = engine_builder or build_retrieval_engine
    # 先读取并校验题目；输入有误会抛给 retrieval_main() 统一处理。
    cases = load_retrieval_cases(args.cases)
    # compare_rerank 同时跑两种配置；否则根据 rerank 只选一种配置。
    variants = (["hybrid", "hybrid_rerank"] if args.compare_rerank
                else ["hybrid_rerank" if args.rerank else "hybrid"])
    # report 是贯穿整个流程的可变字典；results 会随着每题完成逐步追加。
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "evaluation": "retrieval_not_ragas",
              "status": "initializing", "top_k": args.top_k, "case_count": len(cases),
              "variants": variants, "cases_file": str(args.cases), "results": []}
    # 初始化/运行中和失败报告使用独立文件名，保护已有的成功报告。
    running_output = report_prefix(args.output, "running")
    failed_output = report_prefix(args.output, "failed")
    # 尽早落盘初始状态，让长时间加载模型时也能看见当前进度。
    save_retrieval_report(report, running_output, summarize)
    try:
        # 只要任一待测配置需要重排序，就在初始化时请求加载重排序器。
        engine = make_engine(args.project_root, "hybrid_rerank" in variants)
        # 记录业务引擎实际报告的重排序状态，便于解释结果。
        report["rerank_state"] = engine.rerank_state
        # 对比实验要求真实重排序器就绪；不能悄悄退化为普通混合检索。
        if "hybrid_rerank" in variants and (engine.rerank_state != "ready" or engine.reranker is None):
            raise SetupError("精排模型未就绪；未执行伪造的重排对照，请先补齐并加载真实权重")
        # 当前人工标注只用 chunk_index，只有单一来源时块号才不会产生歧义。
        sources = {row["source"] for row in engine.rows}
        if len(sources) != 1:
            raise SetupError("当前标注仅适用于单篇指南；多文档请先改用来源+块号的联合标注")
        # 唯一来源写入报告，便于之后核对语料版本。
        report["knowledge_source"] = next(iter(sources))
        # 收集知识库实际存在的全部块号。
        available = {row["chunk_index"] for row in engine.rows}
        # 集合差非空表示评测标注引用了当前知识库不存在的块。
        if any(set(case["relevant_chunk_indexes"]) - available for case in cases):
            raise SetupError("知识库不包含部分标注块；请核对语料版本，不能用过期标注评测")
    except Exception as error:
        # 初始化失败时不进入逐题循环，也不会产生看似有效的零分。
        report["status"] = "initialization_failed"
        # 项目自定义 SetupError 可展示原文；未知异常经 safe_error 脱敏。
        report["error"] = (str(error) if isinstance(error, SetupError)
                           else "依赖/知识库加载失败：" + safe_error(error))
        save_retrieval_report(report, failed_output, summarize)
        # stderr 供命令行/部署脚本区分错误信息，返回码 1 表示运行失败。
        print(f"{report['error']}；失败报告：{failed_output.with_suffix('.json')}", file=sys.stderr)
        return 1

    # 外层逐题、内层逐配置，使同一道题的对照结果相邻出现。
    for case in cases:
        for variant in variants:
            # 先建立失败默认值；只有完整检索和评分成功后才改为 ok。
            row = {"id": case["id"], "question": case["question"],
                   "relevant_chunk_indexes": case["relevant_chunk_indexes"],
                   "variant": variant, "status": "failed", "scores": None}
            # perf_counter 不受系统时钟校准影响，适合测量一段代码的耗时。
            started = time.perf_counter()
            try:
                # 调用业务检索器；只有 hybrid_rerank 配置传入 use_rerank=True。
                hits = engine.retrieve(case["question"], top_k=args.top_k,
                                       use_rerank=variant == "hybrid_rerank")
                # 若声称执行重排序却没有任何 rerank_score，则拒绝伪造对照结果。
                if variant == "hybrid_rerank" and any("rerank_score" not in hit for hit in hits):
                    raise RuntimeError("rerank_score_missing")
                # 从业务返回结构中抽取块号，保持原有排名顺序。
                row["retrieved_chunk_indexes"] = [hit["entity"]["chunk_index"] for hit in hits]
                # 同时保留来源和两阶段分数，便于排查排序原因；get 允许某阶段分数不存在。
                row["sources"] = [{"source": hit["entity"]["source"],
                                    "chunk_index": hit["entity"]["chunk_index"],
                                    "rrf_score": hit.get("rrf_score"),
                                    "rerank_score": hit.get("rerank_score")} for hit in hits]
                # 用抽出的块号和人工标注计算这一题的各项指标。
                row["scores"] = score_case(
                    row["retrieved_chunk_indexes"], case["relevant_chunk_indexes"], args.top_k)
                # 所有步骤都成功后，才把状态从 failed 改为 ok。
                row["status"] = "ok"
            except Exception as error:
                # 报告只记异常类型，不写可能包含隐私或凭据的异常正文。
                row["error"] = type(error).__name__
            # 无论成功失败都记录耗时，并把本题结果追加到总报告。
            row["elapsed_ms"] = (time.perf_counter() - started) * 1000
            report["results"].append(row)
            # 每完成一个工作项就保存一次，进程意外中止时仍保留已完成进度。
            save_retrieval_report(report, running_output, summarize)
    # 所有题成功才是 completed；只要一题失败就明确标记为带失败完成。
    report["status"] = ("completed" if all(row["status"] == "ok" for row in report["results"])
                        else "completed_with_failures")
    # 全部成功写标准输出；部分失败继续写 failed 文件，不覆盖旧成功结果。
    final_output = args.output if report["status"] == "completed" else failed_output
    save_retrieval_report(report, final_output, summarize)
    print(f"检索报告：{final_output.with_suffix('.json')}；状态={report['status']}")
    # 退出码 0 表示全成功，1 表示至少一项运行失败。
    return 0 if report["status"] == "completed" else 1


def parse_retrieval_args(argv=None, base=None):
    """解析 retrieval 子命令参数并返回 argparse.Namespace。

    argv 为 None 时 argparse 自动读取当前进程命令行；测试可传列表隔离外部参数。
    参数不合法时 argparse 会打印提示并抛出 SystemExit。
    """
    base = Path(base) if base is not None else BASE
    # description 会出现在 --help 输出顶部。
    parser = argparse.ArgumentParser(description="检索指标评测：不生成答案，不调用付费API。")
    # type=Path 把用户输入直接转换成 Path；default 使用项目内的标准文件位置。
    parser.add_argument("--cases", type=Path, default=base / "data/eval/ragas_retrieval_cases.json")
    parser.add_argument("--project-root", type=Path, default=base)
    parser.add_argument("--output", type=Path, default=base / "outputs/retrieval_report")
    # int 类型让 argparse 在遇到非整数时直接报参数错误。
    parser.add_argument("--top-k", type=int, default=4)
    # 互斥组保证“只跑重排序”和“做两种配置对比”不能同时传入。
    group = parser.add_mutually_exclusive_group()
    # store_true 表示选项缺省为 False，命令行出现该开关时变为 True。
    group.add_argument("--compare-rerank", action="store_true")
    group.add_argument("--rerank", action="store_true")
    args = parser.parse_args(argv)
    # 业务上限制 K 的范围，避免无意义的 0 或过大的结果集。
    if not 1 <= args.top_k <= 10:
        parser.error("top-k必须在1到10之间")
    return args


# ---------- RAGAS：默认只校验；--run 才生成答案和调用在线裁判 ----------
def load_ragas_cases(path):
    """读取并规范化 RAGAS 生成质量评测集。

    参数 path 是 JSON 文件路径。返回经过校验的题目字典列表，并把旧字段
    reference_answer 兼容性地统一到 reference。此函数只读本地文件，不调用模型。
    """
    # 与检索评测集相同，utf-8-sig 可兼容带 BOM 的 UTF-8 文件。
    cases = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    # RAGAS 评测必须至少包含一道题，且顶层结构必须是数组。
    if not isinstance(cases, list) or not cases:
        raise ValueError("评测集必须是非空 JSON 数组")
    # ids 用于检查题目编号在整个文件中唯一。
    ids = set()
    for item in cases:
        # 每一项都应是 JSON 对象，对应 Python dict。
        if not isinstance(item, dict):
            raise ValueError("每一题必须是 JSON 对象")
        # id 与 question 都必须存在、是字符串，而且不能只含空白。
        for field in ("id", "question"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"每一题必须包含非空 {field}")
        # 重复 id 会破坏同题配对和结果追踪，因此立即拒绝。
        if item["id"] in ids:
            raise ValueError(f"题目 id 重复：{item['id']}")
        ids.add(item["id"])
        # 与网页输入限制保持一致，避免评测了线上界面无法提交的问题。
        if len(item["question"]) > 500:
            raise ValueError(f"问题超过网页允许的 500 字：{item['id']}")
        # 优先读取新字段 reference；若它不存在，再兼容旧字段 reference_answer。
        reference = item.get("reference", item.get("reference_answer"))
        # 参考答案可以是 None（无人工答案），但若提供就必须是非空字符串。
        if reference is not None and (not isinstance(reference, str) or not reference.strip()):
            raise ValueError(f"reference 必须为非空字符串或 null：{item['id']}")
        # 统一字段名，后续代码无需处理两套格式。
        item["reference"] = reference
        # reference_sources 缺省为空列表；显式提供时也必须是列表。
        if not isinstance(item.get("reference_sources", []), list):
            raise ValueError(f"reference_sources 必须为数组：{item['id']}")
        # 有参考答案却无可追溯出处时，无法验证参考答案是否来自本地知识库。
        if reference and not item.get("reference_sources"):
            raise ValueError(f"参考答案须提供 reference_sources 以便核对：{item['id']}")
    return cases


def validate_references(cases, root):
    """校验参考出处确实存在于项目内的本地文档块。

    cases 是 load_ragas_cases() 返回的题目；root 是项目根目录。返回成功核对的
    摘录条数。这里只验证“路径、块号、原文摘录一致”，不代表医学内容已人工审校。
    """
    # checked 统计验证成功数；cache 避免同一 JSONL 文件被重复读取。
    checked, cache = 0, {}
    for case in cases:
        # 未提供出处时用空列表，因此该题不会进入内层循环。
        for source in case.get("reference_sources", []):
            # <= 在集合间表示“左侧所需键集合是右侧实际键集合的子集”。
            if not isinstance(source, dict) or not {"path", "chunk_index", "quote"} <= source.keys():
                raise ValueError(f"reference_sources 格式错误：{case['id']}")
            # 路径和块号分别要求字符串、整数，避免后面路径拼接或索引出错。
            if not isinstance(source["path"], str) or not isinstance(source["chunk_index"], int):
                raise ValueError("来源path必须为字符串，chunk_index必须为整数")
            # 相对项目根目录拼接路径，再 resolve 消除 .. 和符号链接造成的路径歧义。
            path = (root / source["path"]).resolve()
            # 安全边界：引用文件必须仍位于项目根目录内，不能用 ../ 读取任意文件。
            if not path.is_relative_to(root.resolve()):
                raise ValueError("参考资料路径必须位于项目目录中")
            # 每个来源文件只解析一次，并按知识块 index 建立快速查找字典。
            if path not in cache:
                cache[path] = {row["index"]: row for row in
                               (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                                if line.strip())}
            # dict.get 在目标块号不存在时返回 None，随后给出明确错误。
            row = cache[path].get(source["chunk_index"])
            # quote 必须是真实的非空摘录。
            if not isinstance(source["quote"], str) or not source["quote"].strip():
                raise ValueError("摘录不能为空")
            # 同时验证块存在，且摘录是知识块正文的连续子串。
            if row is None or source["quote"] not in row["text"]:
                raise ValueError(f"参考摘录与本地知识块不符：{case['id']} / {source['chunk_index']}")
            checked += 1
    return checked


def metric_result(value=None, status="ok", reason=None):
    """把单项指标包装成统一、可安全序列化的结果字典。

    value 是原始分数；status 可为 ok、failed、skipped、not_run 等；reason 是解释。
    返回固定包含 value/status/reason 的字典。NaN、无穷和非数字不会被伪装成 0 分。
    """
    # 只有 status=ok 时才要求分数是有限数字；bool 也会被 float 正常转换为 0/1。
    if status == "ok" and (not isinstance(value, (int, float)) or not math.isfinite(value)):
        return {"value": None, "status": "undefined", "reason": "RAGAS 返回非有限值；可能无法提取有效陈述"}
    # 非成功状态统一把 value 设为 None，避免失败或跳过项参与均值。
    return {"value": float(value) if status == "ok" else None, "status": status, "reason": reason}


def safe_error(error):
    """返回脱敏后的异常摘要，只包含异常类名和可选 HTTP 状态码。"""
    # 某些 SDK 异常带 status_code；getattr 在普通异常上安全返回 None。
    status = getattr(error, "status_code", None)
    # 不拼接 str(error)，因为异常正文可能含 API Key、请求内容或服务地址。
    return f"{type(error).__name__}" + (f" (HTTP {status})" if isinstance(status, int) else "")


def aggregate(results):
    """按配置和指标汇总 RAGAS 结果，返回均值及有效样本数。"""
    # summary 的第一层键是检索配置，第二层键是指标名。
    summary = {}
    # 排序保证 JSON/Markdown 输出顺序在多次运行间稳定。
    for variant in sorted({row["variant"] for row in results}):
        rows = [row for row in results if row["variant"] == variant]
        summary[variant] = {}
        for name in RAGAS_METRICS:
            # 取出当前配置下每一题对应的同名指标结果。
            items = [row["metrics"][name] for row in rows]
            # 只有 status=ok 的 value 参与平均；失败、跳过和未定义都会排除。
            valid = [item["value"] for item in items if item["status"] == "ok"]
            summary[variant][name] = {"mean": sum(valid) / len(valid) if valid else None,
                                      "valid": len(valid), "total": len(items),
                                      "excluded": len(items) - len(valid)}
    return summary


def paired_deltas(results):
    """计算同一道题在重排序前后的成对指标差值。

    只纳入 hybrid 与 hybrid_rerank 两边同一指标都成功的题，返回每项指标的
    平均差（重排序后减重排序前）和有效配对数。
    """
    # pairs 形如 {题目id: {配置名: 逐题结果}}。
    pairs = {}
    for row in results:
        # setdefault 在首次看到题目时创建空字典，再按 variant 放入记录。
        pairs.setdefault(row["id"], {})[row["variant"]] = row
    output = {}
    for name in RAGAS_METRICS:
        # differences 收集当前指标每个有效题对的“after - before”。
        differences = []
        for pair in pairs.values():
            # 缺少任一配置就无法做同题配对，因此跳过。
            if not {"hybrid", "hybrid_rerank"} <= pair.keys():
                continue
            # 生成器按给定键顺序依次取出普通检索与重排序结果。
            before, after = (pair[key]["metrics"][name] for key in ("hybrid", "hybrid_rerank"))
            # 链式比较等价于 before[status]==ok 且 after[status]==ok。
            if before["status"] == after["status"] == "ok":
                differences.append(after["value"] - before["value"])
        # 没有有效配对时均值为 None，避免把“无数据”写成 0 改进。
        output[name] = {"mean_delta": sum(differences) / len(differences) if differences else None,
                        "paired_valid": len(differences)}
    return output


def package_versions():
    """查询关键依赖版本，返回“包名 -> 版本或 None”的字典。"""
    versions = {}
    # 报告这些包可以帮助判断评测能否复现，以及 API 是否与固定代码兼容。
    for name in ("ragas", "langchain", "langchain-core", "langchain-openai", "openai",
                 "sentence-transformers", "pymilvus"):
        try:
            # metadata.version 只读已安装包元数据，不导入这些较重的库。
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            # 未安装记为 None，让默认输入校验模式仍可运行。
            versions[name] = None
    return versions


def ragas_markdown(report):
    """把 RAGAS 报告字典渲染为 Markdown 文本并返回。"""
    # 报告开头说明状态、进度及均值计算口径。
    lines = ["# RAGAS 生成质量评测", "", f"状态：{report['status']}", f"时间：{report['created_at']}",
             f"已建立结果记录（含待执行）：{len(report['results'])}/{report['planned_rows']}", "",
             "验证模式不会产生分数；缺少参考答案时 precision/recall 为 skipped；无检索资料时 faithfulness/precision 为 skipped。",
             "失败、未定义和跳过不按 0 分参与均值。裁判默认复用回答模型，但可用参数或评测环境变量覆盖；自动裁判可能有偏差，小样本不能代表临床可靠性。", "",
             "| 配置 | 指标 | 均值 | 有效/总数 |", "|---|---|---:|---:|"]
    # 展开“配置 -> 指标”两层汇总，每项指标占表格一行。
    for variant, metrics in report["summary"].items():
        for name, item in metrics.items():
            # None 表示没有任何有效样本，因此显示“未产生”而不是 0.0000。
            score = "未产生" if item["mean"] is None else f"{item['mean']:.4f}"
            lines.append(f"| {variant} | {name} | {score} | {item['valid']}/{item['total']} |")
    # paired_deltas 和 metadata 用格式化 JSON 保留完整嵌套结构。
    lines.extend(["", "## 同题重排序对照", "",
                  "只用两边均成功的题目计算 rerank − hybrid；正负变化均如实记录。", "",
                  "```json", json.dumps(report["paired_deltas"], ensure_ascii=False, indent=2), "```", "",
                  "## 配置与版本", "", "```json",
                  json.dumps(report["metadata"], ensure_ascii=False, indent=2), "```", "",
                  "## 每题状态", "", "| ID | 配置 | 状态 | 指标状态 |", "|---|---|---|---|"])
    # 逐题状态表只放简洁状态，完整文本与错误仍保存在同名 JSON。
    for row in report["results"]:
        # 把四个指标状态拼成一个单元格，例如 faithfulness: ok; ...。
        states = "; ".join(f"{name}: {item['status']}" for name, item in row["metrics"].items())
        lines.append(f"| {row['id']} | {row['variant']} | {row['status']} | {states} |")
    lines.extend(["", "完整 question / answer / contexts / reference / 来源与逐项错误见同名 JSON。", ""])
    # 列表按换行符合并；最后一个空字符串会让文档以换行结尾。
    return "\n".join(lines)


def save_ragas_report(report, prefix):
    """重新汇总并原子写入 JSON、Markdown 两份 RAGAS 报告。"""
    # 每次落盘都以当前 results 重新计算总览和成对差值，适用于运行中快照。
    report["summary"] = aggregate(report["results"])
    report["paired_deltas"] = paired_deltas(report["results"])
    # prefix 是不带扩展名的路径；先确保父目录存在。
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False 保留中文，indent=2 方便阅读，allow_nan=False 保证标准 JSON。
    contents = ((".json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)),
                (".md", ragas_markdown(report)))
    for suffix, body in contents:
        # 与检索报告一样，先写临时文件再替换正式文件，减少半写入状态。
        target = prefix.with_suffix(suffix)
        temporary = target.with_suffix(suffix + ".tmp")
        temporary.write_text(body, encoding="utf-8")
        temporary.replace(target)


def build_metrics(engine, args, usage):
    """创建四个 RAGAS 指标对象。

    参数：
        engine: 已初始化的业务 RAG 对象，提供本地嵌入模型和回答模型配置。
        args: parse_ragas_args() 返回的命令行参数。
        usage: 报告中的可变用量字典，回调会原地累加裁判调用与 Token 数。
    返回：以 RAGAS_METRICS 中名称为键、指标实例为值的字典。

    此函数只会在显式 --run 后调用；裁判复用 OpenAI 兼容接口，本地 BGE 负责向量计算。
    """
    # 这些依赖较重且只属于真实评分流程，因此放在函数内延迟导入。
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import BaseRagasEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference,
                               LLMContextRecall, ResponseRelevancy)
    from ragas.run_config import RunConfig

    # 把项目现有的 BGE embedder 适配为 RAGAS 所需的嵌入接口，避免另建在线嵌入服务。
    class LocalBGE(BaseRagasEmbeddings):
        def embed_documents(self, texts):
            """同步编码多段文本，返回 RAGAS 需要的普通二维列表。"""
            # normalize_embeddings=True 生成单位向量，适合用点积/余弦相似度比较。
            return engine.embedder.encode(texts, normalize_embeddings=True).tolist()

        def embed_query(self, text):
            """同步编码单条查询，返回一维向量。"""
            # 复用批量实现，传入单元素列表后取第一个向量。
            return self.embed_documents([text])[0]

        async def aembed_documents(self, texts):
            """提供 RAGAS 要求的异步批量接口；底层仍调用本地同步模型。"""
            return self.embed_documents(texts)

        async def aembed_query(self, text):
            """提供单条查询的异步接口。"""
            return self.embed_query(text)

    # LangChain 回调：每次裁判模型成功结束时，把返回的 Token 用量累加进报告。
    class TokenCounter(BaseCallbackHandler):
        # run_inline=True 要求回调在当前执行路径内运行，避免报告保存时计数尚未完成。
        run_inline = True

        def on_llm_end(self, response, **kwargs):
            """接收 LangChain 的模型结束事件；无返回值。"""
            # 一次 RAGAS 指标可能触发多次裁判调用，这里按实际完成调用计数。
            usage["completed_evaluator_calls"] += 1
            # 不同兼容服务可能不返回 llm_output 或 token_usage，因此都提供空字典后备值。
            tokens = (response.llm_output or {}).get("token_usage") or {}
            # 三类 Token 分开累计；缺失、None 等值按 0 处理并转为整数。
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[field] += int(tokens.get(field, 0) or 0)
            # 单独统计缺失用量的调用，提醒读者报告中的 Token 合计可能不完整。
            if not tokens:
                usage["calls_without_token_usage"] += 1

    # RunConfig 控制单项指标的超时、尝试次数、退避等待、并发量和随机种子。
    # 这里的 +1 把用户允许的“额外重试次数”转换为包含首次请求在内的尝试上限。
    run_config = RunConfig(timeout=args.timeout, max_retries=args.max_retries + 1,
                           max_wait=5, max_workers=args.concurrency, seed=42)
    # ChatOpenAI 也支持 DeepSeek 等 OpenAI 兼容服务；评测专用环境变量优先于回答模型配置。
    judge = ChatOpenAI(model=args.judge_model or engine.model, temperature=0,
                       api_key=os.environ.get("EVAL_API_KEY") or os.environ["LLM_API_KEY"],
                       base_url=os.environ.get("EVAL_BASE_URL") or str(engine.llm.base_url),
                       max_tokens=args.judge_max_tokens, timeout=args.timeout, max_retries=0,
                       extra_body=engine.options.get("extra_body", {}), callbacks=[TokenCounter()])
    # LangchainLLMWrapper 把 LangChain 聊天模型适配给 RAGAS；LocalBGE 负责相关性向量。
    llm, embeddings = LangchainLLMWrapper(judge, run_config=run_config), LocalBGE()
    # 嵌入适配器也接收同一运行配置，使超时/并发等设置保持一致。
    embeddings.set_run_config(run_config)
    # 四个对象分别衡量忠实度、答案相关性、上下文精度和上下文召回率。
    # strictness=1 让相关性指标只生成一条反向问题，限制裁判调用与成本。
    metrics = {"faithfulness": Faithfulness(llm=llm),
               "answer_relevancy": ResponseRelevancy(llm=llm, embeddings=embeddings, strictness=1),
               "context_precision": LLMContextPrecisionWithReference(llm=llm),
               "context_recall": LLMContextRecall(llm=llm)}
    # 逐个复制并调整指标提示词，避免直接改动库对象可能共享的原始提示词。
    for metric in metrics.values():
        prompts = {}
        for name, original in metric.get_prompts().items():
            prompt = copy.deepcopy(original)
            # 答案相关性会根据回答反向生成问题；要求它与回答使用同一种语言。
            if metric is metrics["answer_relevancy"]:
                prompt.instruction += " Generate the question in the same language as the response."

            # RAGAS 的提示词解析失败时可能自动请求模型修复 JSON；评测禁用这种隐藏调用。
            # 默认参数 _generate=... 会在定义函数时固定当前提示词的原方法，避免闭包晚绑定。
            async def without_repairs(*call_args, _generate=prompt.generate_multiple, **call_kwargs):
                # retries_left=0 明确禁止解析失败后的额外模型修复请求。
                call_kwargs["retries_left"] = 0
                return await _generate(*call_args, **call_kwargs)

            # 用包装函数替换当前提示词实例的方法，再按原提示词名称保存。
            prompt.generate_multiple = without_repairs
            prompts[name] = prompt
        # 把复制后的提示词装回指标，并用统一 RunConfig 完成初始化。
        metric.set_prompts(**prompts)
        metric.init(run_config)
    return metrics


async def score_row(row, metrics, args):
    """异步计算一条已生成回答的所有适用指标，并原地更新 row。

    row 包含 question、answer、contexts、reference 等字段；metrics 是 build_metrics()
    返回的字典；args 提供超时和并发限制。本函数不返回新对象，结果写入 row。
    """
    # 延迟导入确保默认“仅校验”模式不要求安装 RAGAS。
    from ragas import SingleTurnSample
    # SingleTurnSample 是 RAGAS 对单轮问答的标准输入结构。
    sample = SingleTurnSample(user_input=row["question"], response=row["answer"],
                              retrieved_contexts=row["contexts"], reference=row["reference"])
    # 信号量限制这一行样本中同时运行的指标数，防止瞬时请求过多。
    semaphore = asyncio.Semaphore(args.concurrency)

    async def score(name, metric):
        """计算一个指标，返回 (指标名, 统一结果字典) 二元组。"""
        # 上下文精度/召回率依赖人工参考答案；缺失时明确跳过。
        if name in {"context_precision", "context_recall"} and not row["reference"]:
            return name, metric_result(status="skipped", reason="没有人工参考答案")
        # 忠实度与上下文精度需要检索资料；空上下文时无法计算。
        if name in {"faithfulness", "context_precision"} and not row["contexts"]:
            return name, metric_result(status="skipped", reason="没有召回资料，无法判断回答是否忠于资料/资料精度")
        # async with 会等待并发名额，并在离开代码块时自动归还。
        async with semaphore:
            try:
                # single_turn_ascore 调用该指标的异步单轮评分 API，并传入单项超时。
                value = await metric.single_turn_ascore(sample, timeout=args.timeout)
                # 转为 float 后再交给 metric_result 检查 NaN/无穷等异常值。
                return name, metric_result(float(value))
            except Exception as error:
                # 单项失败不会取消其他指标；只记录脱敏后的失败原因。
                return name, metric_result(status="failed", reason=safe_error(error))

    # 为每个指标创建协程并并发等待；二元组序列可直接转换成字典。
    row["metrics"] = dict(await asyncio.gather(*(score(name, metric) for name, metric in metrics.items())))
    # 任一指标 failed/undefined 就标记 partial_failure；正常跳过仍算评测流程完成。
    row["status"] = ("partial_failure" if any(value["status"] in {"failed", "undefined"}
                                                for value in row["metrics"].values()) else "evaluated")


def new_row(case, variant):
    """根据题目和配置名创建一条尚未运行的 RAGAS 结果记录。"""
    # answer/contexts/sources 先放空值；四个指标都用 not_run 明确表示尚未执行。
    return {"id": case["id"], "question": case["question"], "reference": case["reference"],
            "reference_sources": case.get("reference_sources", []), "variant": variant,
            "answer": None, "contexts": [], "sources": [], "status": "pending",
            "metrics": {name: metric_result(status="not_run") for name in RAGAS_METRICS}}


def run_ragas(args, expected_ragas=None, package_versions_function=None,
              build_metrics_function=None):
    """执行 RAGAS 输入校验，或在显式授权后执行真实生成质量评测。

    参数 args 来自 parse_ragas_args()。默认 args.run=False，只读取并核对本地文件；
    只有 args.run=True 才会初始化回答模型和裁判。返回 0 表示成功，1 表示部分
    生成/评分失败，2 表示固定版本或环境初始化失败。输入/文件错误会抛给
    ragas_main()，由它转换成退出码2和可读提示。
    """
    # 入口显式传入可替换依赖，避免通过globals/setattr偷偷修改模块状态。
    expected_ragas = expected_ragas or EXPECTED_RAGAS
    versions_reader = package_versions_function or package_versions
    metrics_builder = build_metrics_function or build_metrics
    # 先加载全部题目，再按 --limit 截取前 N 题；切片不会修改原列表。
    cases = load_ragas_cases(args.cases)[:args.limit]
    # 逐条核对参考摘录与本地知识块，并取得通过核对的摘录数量。
    verified = validate_references(cases, args.project_root)
    # compare_rerank 做同题双配置对照；否则默认测重排序，--no-rerank 改测普通混合检索。
    variants = (["hybrid", "hybrid_rerank"] if args.compare_rerank
                else ["hybrid" if args.no_rerank else "hybrid_rerank"])
    # 初始化报告。默认状态 validated_only 是安全门控：尚未生成答案或评分。
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "validated_only",
              # planned_rows 等于题目数乘配置数；results 会在真实运行时逐项填充。
              "planned_rows": len(cases) * len(variants), "results": [], "summary": {}, "paired_deltas": {},
              # metadata 记录复现实验所需的版本、参数、策略和已知限制。
              "metadata": {"packages": versions_reader(),
                            # 评测集原始字节的 SHA-256 可确认后来使用的是否为同一份文件。
                            "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                            "verified_reference_excerpts": verified, "top_k": args.top_k,
                            "case_limit": args.limit, "variants": variants, "concurrency": args.concurrency,
                            "max_retries": args.max_retries, "judge_max_tokens": args.judge_max_tokens,
                            "timeout": args.timeout, "answer_relevancy_strictness": 1,
                            # 评测固定使用空历史且不写 Redis，避免真实用户对话影响结果。
                            "history": "empty; no Redis write",
                            # 显式记录人类可读指标名与实际 RAGAS 类的对应关系。
                            "ragas_metric_classes": {"faithfulness": "Faithfulness",
                                                     "answer_relevancy": "ResponseRelevancy",
                                                     "context_precision": "LLMContextPrecisionWithReference",
                                                     "context_recall": "LLMContextRecall"},
                            # 记录 JSON 修复、回答温度和各种缺失输入的处理口径。
                            "json_repair_retries": 0, "answer_temperature": 0.3,
                            "metric_policy": {"context_precision": "reference-based average precision",
                                              "missing_reference": "skip precision and recall",
                                              "empty_contexts": "skip faithfulness and precision; recall still eligible",
                                              "relevancy_question_language": "same as response",
                                              "retry_scope": "RAGAS runtime retries; SDK and JSON repair retries disabled"},
                            # 这些警告限制对分数的解释范围，避免把自动小样本评测当作临床结论。
                            "warnings": ["少量离线资料题的评测集，非临床效果证明",
                                         "自动裁判可能产生偏差，需人工抽检"],
                            # TokenCounter 会在真实评分时原地累加以下字段。
                            "evaluator_usage": {"completed_evaluator_calls": 0,
                                                "calls_without_token_usage": 0, "prompt_tokens": 0,
                                                "completion_tokens": 0, "total_tokens": 0}}}
    # 无论是否真实运行，都先保存一份校验结果，留下可审计的输入快照。
    save_ragas_report(report, args.output)
    # 这是付费边界：没有显式 --run 就在此返回，不导入 RAG、不生成答案、不创建裁判。
    if not args.run:
        print(f"已校验 {len(cases)} 题 / {verified} 条出处；没有调用模型，没有生成 RAGAS 分数。")
        return 0
    # 真实评分只接受固定版本；版本不一致时保存错误后退出，避免产生不可比结果。
    if report["metadata"]["packages"]["ragas"] != expected_ragas:
        report["status"] = "setup_failed"
        report["error"] = "请安装 requirements-eval.txt 中固定的 RAGAS 0.2.15 依赖组"
        save_ragas_report(report, args.output)
        print(report["error"], file=sys.stderr)
        return 2
    try:
        # 将待测项目放到模块搜索路径首位，然后关闭 RAGAS 遥测和 Hugging Face 联网下载。
        sys.path.insert(0, str(args.project_root))
        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        # 仅通过 --run 门控后才延迟导入和创建真正的 RAG 引擎。
        from app.single_app import RAG
        # with_memory=False 保证评测不读取或写入真实 Redis 对话历史。
        engine = RAG(with_memory=False)
        # 待测配置需要重排序时必须确认模型已实际加载，禁止静默降级。
        if "hybrid_rerank" in variants and engine.reranker is None:
            raise SetupError("重排序模型未加载；请启用重排序，或用 --no-rerank 仅测混合检索")
        # 创建四个真实指标，并把 usage 字典交给回调原地更新。
        metrics = metrics_builder(engine, args, report["metadata"]["evaluator_usage"])
        # 优先记录评测专用 URL；urlsplit 将其拆成组成部分以便移除敏感内容。
        base_url = urlsplit(os.environ.get("EVAL_BASE_URL") or str(engine.llm.base_url))
        # 把实际使用的回答、裁判、嵌入和重排序配置写入报告。
        report["metadata"].update({"answer_model": engine.model,
                                   "judge_model": args.judge_model or engine.model,
                                   # 只保留 scheme、hostname 和 path，不记录账号、密码、查询串或 fragment。
                                   "judge_base_url": f"{base_url.scheme}://{base_url.hostname}{base_url.path}",
                                   "embedding_model": os.environ.get("RAG_EMBED_MODEL", "bge-small-zh-v1.5"),
                                   "reranker_model": os.environ.get("RAG_RERANK_MODEL", "bge-reranker-base"),
                                   "reranker_loaded": engine.reranker is not None,
                                   "min_vector_score": os.environ.get("RAG_MIN_VECTOR_SCORE", "0.55"),
                                   "thinking": engine.options.get("extra_body", {}).get("thinking"),
                                   "answer_max_tokens": os.environ.get("LLM_MAX_TOKENS", "768")})
        # 若核心业务文件存在，则记录其哈希，帮助判断报告由哪版回答逻辑产生。
        core_path = args.project_root / "app/single_app.py"
        if core_path.exists():
            report["metadata"]["core_sha256"] = hashlib.sha256(core_path.read_bytes()).hexdigest()
    except Exception as error:
        # 初始化失败不会进入答案生成；保存清晰状态并返回专用退出码 2。
        report["status"] = "setup_failed"
        # 可预期配置错误展示自定义消息，未知异常只记录脱敏类型。
        report["error"] = str(error) if isinstance(error, SetupError) else safe_error(error)
        save_ragas_report(report, args.output)
        print(f"初始化失败：{report['error']}；请核对依赖、模型路径及数据库服务。", file=sys.stderr)
        return 2
    # 初始化完成后才进入 running 状态。
    report["status"] = "running"
    # work_items 保存题目、配置和对应的可变结果字典，供异步内部函数逐项处理。
    work_items = []
    for case in cases:
        for variant in variants:
            # 预先创建所有 pending 行，使首份 running 快照能显示完整计划。
            row = new_row(case, variant)
            report["results"].append(row)
            work_items.append((case, variant, row))
    # 在任何真实模型调用前落盘 running 快照。
    save_ragas_report(report, args.output)

    async def evaluate_cases():
        """按工作项顺序完成检索、答案生成与异步指标评分。"""
        # 工作项顺序执行，便于持续保存进度；每行内部的指标可由 score_row 并发。
        for case, variant, row in work_items:
            try:
                # 获取前 K 条知识块；配置名决定是否调用真实重排序器。
                hits = engine.retrieve(case["question"], top_k=args.top_k,
                                       use_rerank=variant == "hybrid_rerank")
                # RAGAS 只需要知识块正文作为 retrieved_contexts。
                row["contexts"] = [hit["entity"]["text"] for hit in hits]
                # 报告额外保留来源、块号和排序分数，用于人工追踪。
                row["sources"] = [{"source": hit["entity"].get("source"),
                                   "chunk_index": hit["entity"].get("chunk_index"),
                                   "rrf_score": hit.get("rrf_score"),
                                   "rerank_score": hit.get("rerank_score")} for hit in hits]
                # 固定传入“无历史对话”，避免多轮记忆给不同题目带来污染。
                row["answer"] = engine.generate(case["question"], "（无历史对话）", hits)
                # 生成结果必须是非空字符串，否则按生成失败处理。
                if not row["answer"] or not isinstance(row["answer"], str):
                    raise ValueError("模型没有返回答案")
                # 先标记 generated 并保存：即使后续评分中断，也能保留已经生成的答案。
                row["status"] = "generated"
                save_ragas_report(report, args.output)
                # 计算此行所有适用指标；函数会把 row 更新为 evaluated 或 partial_failure。
                await score_row(row, metrics, args)
            except Exception as error:
                # 已成功生成后的异常属于评分失败，否则属于检索/生成失败。
                stage = "evaluation_failed" if row["status"] == "generated" else "generation_failed"
                # 多重赋值同时写状态和脱敏错误摘要。
                row["status"], row["error"] = stage, safe_error(error)
                # 整行流程异常时四个指标均未完整运行，统一标记 not_run 并记录阶段。
                row["metrics"] = {name: metric_result(status="not_run", reason=stage)
                                  for name in RAGAS_METRICS}
            # 每个工作项结束都保存报告；即使失败也不会丢失之前成功的结果。
            save_ragas_report(report, args.output)
            print(f"{row['id']} / {variant}: {row['status']}")

    # 在同步命令行入口创建并运行事件循环，直到全部工作项处理完毕。
    asyncio.run(evaluate_cases())
    # 只有严格等于 evaluated 的行才视为成功；partial_failure 等都使总报告带失败。
    failed = any(row["status"] != "evaluated" for row in report["results"])
    report["status"] = "completed_with_failures" if failed else "completed"
    # 保存最终汇总并提示 JSON 报告位置及 Token 统计口径。
    save_ragas_report(report, args.output)
    print(f"报告已保存：{args.output.with_suffix('.json')}（Token 为裁判返回的统计，不含回答生成；不折算金额）")
    # 全部成功返回 0；任何行失败返回 1，便于 CI 或脚本检测。
    return 1 if failed else 0


def parse_ragas_args(argv=None, base=None):
    """解析 ragas 子命令参数，校验数值范围并选择默认输出路径。

    argv 为 None 时读取当前进程命令行；测试可以传入字符串列表。返回 argparse.Namespace。
    参数格式或范围错误时，argparse 会打印说明并抛出 SystemExit。
    """
    base = Path(base) if base is not None else BASE
    # help 中明确默认行为与付费边界，避免用户误以为普通命令会产生分数。
    parser = argparse.ArgumentParser(description="真实RAGAS评测：默认只校验；--run才调用付费模型。")
    # --cases：生成评测题 JSON 文件；type=Path 自动转换成 Path 对象。
    parser.add_argument("--cases", type=Path, default=base / "data/eval/ragas_generation_cases.json")
    # --project-root：被测项目根目录，用于导入 app 并限制参考资料读取范围。
    parser.add_argument("--project-root", type=Path, default=base)
    # --output：报告路径前缀，不写扩展名；None 表示稍后按是否 --run 自动选择。
    parser.add_argument("--output", type=Path, default=None)
    # --run：唯一的真实模型调用开关；未提供时只校验本地输入。
    parser.add_argument("--run", action="store_true", help="实际调用生成模型和RAGAS裁判，会消耗API额度")
    # --limit：只评测输入文件前 N 题，默认 2，可控制测试范围和费用。
    parser.add_argument("--limit", type=int, default=2)
    # --top-k：每道题交给回答模型和指标的检索知识块数量。
    parser.add_argument("--top-k", type=int, default=4)
    # --concurrency：一条样本中最多同时执行多少个异步指标。
    parser.add_argument("--concurrency", type=int, default=1)
    # --max-retries：RAGAS运行层允许的额外重试次数；异常范围由固定版本RunConfig决定。
    parser.add_argument("--max-retries", type=int, default=0)
    # --timeout：单次模型/指标调用的超时秒数。
    parser.add_argument("--timeout", type=int, default=120)
    # --judge-model：可单独指定裁判模型；None 时复用回答模型名称。
    parser.add_argument("--judge-model", default=None)
    # --judge-max-tokens：裁判单次回复允许生成的最大 Token 数。
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    # 两个检索配置开关互斥：一个做双配置对比，一个仅测无重排序配置。
    group = parser.add_mutually_exclusive_group()
    # --compare-rerank：同题分别运行 hybrid 和 hybrid_rerank，计算成对差值。
    group.add_argument("--compare-rerank", action="store_true")
    # --no-rerank：只测 hybrid；两者都不传时默认只测 hybrid_rerank。
    group.add_argument("--no-rerank", action="store_true")
    # 实际解析参数；argv=None 时 argparse 自动读取 sys.argv[1:]。
    args = parser.parse_args(argv)
    # 校验模式与真实运行使用不同默认文件，防止一次普通校验覆盖付费生成报告。
    if args.output is None:
        filename = "generation_report" if args.run else "input_validation"
        args.output = base / "outputs/ragas" / filename
    # 每个三元组依次是属性名、允许最小值、允许最大值。
    limits = (("limit", 1, 50), ("top_k", 1, 10), ("concurrency", 1, 4),
              ("max_retries", 0, 2), ("timeout", 10, 300), ("judge_max_tokens", 256, 4096))
    for name, minimum, maximum in limits:
        # getattr 按字符串读取 Namespace 属性，让多个参数共用同一校验逻辑。
        if not minimum <= getattr(args, name) <= maximum:
            # parser.error 会打印 usage 和中文错误并以命令行参数错误结束。
            parser.error(f"{name} 必须在 {minimum} 到 {maximum} 之间")
    return args


def retrieval_main(argv=None):
    """retrieval 子命令入口：返回评测退出码，并把常见输入错误转换为退出码 2。"""
    try:
        # 先解析参数，再执行检索评测；正常退出码由 run_retrieval 决定。
        return run_retrieval(parse_retrieval_args(argv))
    except (ValueError, OSError, json.JSONDecodeError) as error:
        # 捕获数据格式、文件系统和 JSON 解析错误，向 stderr 输出可操作提示。
        print("评测输入错误：" + str(error), file=sys.stderr)
        return 2


def ragas_main(argv=None):
    """ragas 子命令入口：返回评测退出码，并把本地输入/文件错误转换为 2。"""
    try:
        # 默认只会运行安全的输入校验；是否真实调用由解析出的 args.run 决定。
        return run_ragas(parse_ragas_args(argv))
    except (ValueError, OSError, json.JSONDecodeError) as error:
        # 初始化后的模型调用异常由 run_ragas 自己记录；这里只处理本地输入错误。
        print(f"输入/文件错误：{error}", file=sys.stderr)
        return 2


def main(argv=None):
    """顶层命令分发器，根据第一个位置参数调用 retrieval 或 ragas。

    argv=None 时读取 sys.argv[1:]；传列表便于单元测试。返回对应子命令的整数退出码。
    """
    # 复制成新列表，既统一处理元组等可迭代对象，也避免修改调用者传入的对象。
    arguments = list(sys.argv[1:] if argv is None else argv)
    # __doc__ 使用文件顶部说明作为总命令帮助文本。
    parser = argparse.ArgumentParser(description=__doc__)
    # command 只能二选一；choices 会让其他值自动得到清晰的参数错误。
    parser.add_argument("command", choices=("retrieval", "ragas"), help="retrieval测检索；ragas测生成质量")
    # 没有任何参数时主动显示帮助，并用 2 表示命令行用法错误。
    if not arguments:
        parser.print_help()
        return 2
    # 这里只解析第一个词以确定子命令，剩余参数交给各自的解析器。
    command = parser.parse_args(arguments[:1]).command
    # 条件表达式选择入口；arguments[1:] 去掉已经消费的子命令名。
    return retrieval_main(arguments[1:]) if command == "retrieval" else ragas_main(arguments[1:])


if __name__ == "__main__":
    # 只有直接执行 python -m app.evaluate 时才运行；被测试导入时不会自动评测。
    # SystemExit 把 main() 返回的整数交给操作系统作为进程退出码。
    raise SystemExit(main())
