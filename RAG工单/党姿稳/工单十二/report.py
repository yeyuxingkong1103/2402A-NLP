# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-LightRAG 优化任务
# 模块职责：产出评估交付物——知识图谱统计与可视化、检索结果对比、RAGAS 指标对比文档

import os                                                       # 导入 os 用于路径拼接
import json                                                     # 导入 json 用于读取评估结果
import collections                                              # 导入 collections 用于频次统计
import config                                                   # 导入本工单配置模块
from logger import LOGGER                                       # 导入统一日志对象


def _md_table(headers, rows):                                   # 定义 Markdown 表格渲染函数
    """把表头与数据行渲染为 Markdown 表格文本。"""
    out = ["| " + " | ".join(headers) + " |",                   # 表头行
           "|" + "|".join(["---"] * len(headers)) + "|"]        # 分隔行
    for r in rows:                                              # 遍历数据行
        out.append("| " + " | ".join(str(x) for x in r) + " |")  # 渲染该行
    return "\n".join(out)                                       # 拼接为完整表格


def _write(path, text):                                         # 定义文件写入函数
    """以 UTF-8 写入文本文件并输出日志。"""
    config.ensure_dirs()                                        # 确保输出目录存在
    with open(path, "w", encoding="utf-8") as f:                # 以 UTF-8 打开文件
        f.write(text)                                           # 写入内容
    LOGGER.info("已生成 %s", path)                               # 输出生成日志


# ---------------------------------------------------------------------------
# 一、知识图谱统计与可视化
# ---------------------------------------------------------------------------
def _entity_types(graph):                                       # 定义实体类型统计函数
    """统计各类实体的数量，按数量降序返回。"""
    cnt = collections.Counter(r["type"] or "未分类" for r in graph["entities"].values())  # 统计类型频次
    return cnt.most_common()                                    # 返回降序列表


def _relation_types(graph):                                     # 定义关系类型统计函数
    """统计各类关系的数量，按数量降序返回。"""
    cnt = collections.Counter(r["relation"] for r in graph["relations"])  # 统计关系频次
    return cnt.most_common()                                    # 返回降序列表


def _degrees(graph):                                            # 定义实体度数统计函数
    """统计每个实体参与的关系统数（度），用于挑选图谱中心节点。"""
    deg = collections.Counter()                                 # 初始化度数计数器
    for r in graph["relations"]:                                # 遍历全部关系
        deg[r["head"]] += 1                                     # 头实体度数加一
        deg[r["tail"]] += 1                                     # 尾实体度数加一
    return deg                                                  # 返回度数计数器


def draw_graph(graph, top_n=60, path=None):                     # 定义图谱可视化函数
    """取度数最高的若干实体与其间关系绘制网络图，输出 PNG。"""
    import matplotlib                                           # 延迟导入绘图库（仅画图时需要）
    matplotlib.use("Agg")                                       # 使用无界面后端，服务器环境也能出图
    import matplotlib.pyplot as plt                             # 导入绘图接口
    import networkx as nx                                       # 导入图算法库
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]  # 设置中文字体，避免乱码
    plt.rcParams["axes.unicode_minus"] = False                  # 正常显示负号
    deg = _degrees(graph)                                       # 计算实体度数
    top = [n for n, _ in deg.most_common(top_n)]                # 取度数最高的实体
    keep = set(top)                                             # 转集合便于判定
    g = nx.Graph()                                              # 创建无向图
    for n in top:                                               # 加入节点并记录类型
        rec = graph["entities"].get(n, {})                       # 取实体记录
        g.add_node(n, etype=rec.get("type") or "未分类")          # 记录节点类型用于着色
    for r in graph["relations"]:                                # 加入节点之间的关系
        if r["head"] in keep and r["tail"] in keep:             # 两端都在保留集合内才加入
            g.add_edge(r["head"], r["tail"], label=r["relation"])  # 记录关系类型
    if g.number_of_nodes() == 0:                                # 空图时跳过绘制
        LOGGER.warning("图谱为空，跳过可视化")                    # 输出告警
        return None                                             # 返回 None
    types = sorted({d["etype"] for _, d in g.nodes(data=True)})  # 收集全部实体类型
    palette = plt.get_cmap("tab20")                             # 取 20 色色板
    color_of = {t: palette(i % 20) for i, t in enumerate(types)}  # 为每种类型分配颜色
    colors = [color_of[d["etype"]] for _, d in g.nodes(data=True)]  # 按节点类型取颜色
    sizes = [300 + 120 * deg[n] for n in g.nodes()]             # 节点大小随度数增大
    plt.figure(figsize=(20, 15))                                # 创建大画布
    pos = nx.spring_layout(g, seed=42, k=0.9, iterations=120)   # 力导向布局（固定随机种子保证可复现）
    nx.draw_networkx_edges(g, pos, alpha=0.25, width=0.8)       # 绘制边（淡化避免遮挡）
    nx.draw_networkx_nodes(g, pos, node_color=colors, node_size=sizes,  # 绘制节点
                           edgecolors="white", linewidths=0.6)  # 节点描白边提升可读性
    nx.draw_networkx_labels(g, pos, font_size=8, font_family="Microsoft YaHei")  # 绘制实体名标签
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=10,  # 构造图例句柄
                          color=color_of[t], label=t) for t in types]  # 每种类型一个图例项
    plt.legend(handles=handles, loc="upper right", fontsize=11, title="实体类型")  # 绘制图例
    plt.title("招股说明书知识图谱（度数 Top %d 实体）" % g.number_of_nodes(), fontsize=18)  # 图标题
    plt.axis("off")                                             # 关闭坐标轴
    plt.tight_layout()                                          # 自动调整边距
    path = path or os.path.join(config.EVAL_DIR, "knowledge_graph.png")  # 默认输出路径
    plt.savefig(path, dpi=150, bbox_inches="tight")             # 保存为 PNG
    plt.close()                                                 # 关闭画布释放内存
    LOGGER.info("已生成 %s（%d 节点 %d 边）", path, g.number_of_nodes(), g.number_of_edges())  # 输出日志
    return path                                                 # 返回图片路径


def export_graph_report(graph):                                 # 定义图谱文档生成函数
    """生成知识图谱统计文档（实体类型分布、关系类型分布、核心实体与样例关系）。"""
    deg = _degrees(graph)                                       # 计算实体度数
    lines = ["# 知识图谱统计", ""]                                # 初始化文档行
    lines += ["- 语料块总数：%d（参与建图 %d 块）" % (graph["stats"]["chunk_total"],
                                                    min(config.GRAPH_BUILD_LIMIT, graph["stats"]["chunk_total"]))]  # 语料规模
    lines += ["- 实体总数：%d" % len(graph["entities"]),          # 实体总数
              "- 关系总数：%d（按三元组去重后）" % len(graph["relations"]),  # 关系总数
              "- 实体类型数：%d" % len(_entity_types(graph)),      # 实体类型数
              "- 关系类型数：%d" % len(_relation_types(graph)), ""]  # 关系类型数
    lines += ["## 一、实体类型分布（schema 定义 12 类）", ""]     # 实体类型小节
    lines += [_md_table(["实体类型", "数量"], _entity_types(graph)), ""]  # 实体类型表
    lines += ["## 二、关系类型分布（schema 定义 22 类）", ""]     # 关系类型小节
    lines += [_md_table(["关系类型", "数量"], _relation_types(graph)), ""]  # 关系类型表
    lines += ["## 三、核心实体（按关系统数 Top 20）", ""]         # 核心实体小节
    rows = []                                                   # 核心实体表行
    for n, d in deg.most_common(20):                            # 取度数最高的实体
        rec = graph["entities"].get(n, {})                       # 取实体记录
        rows.append([n, rec.get("type", ""), d, (rec.get("desc") or "")[:40]])  # 名称、类型、度数、描述
    lines += [_md_table(["实体", "类型", "关系统数", "描述"], rows), ""]  # 核心实体表
    lines += ["## 四、高置信关系样例（按支撑次数 Top 25）", ""]   # 关系样例小节
    rows = []                                                   # 关系表行
    for r in graph["relations"][:25]:                           # 取支撑次数最高的关系
        val = "、".join(r["values"][:3]) if r["values"] else ""   # 拼接关系取值
        rows.append([r["head"], r["relation"], r["tail"], val, r["weight"]])  # 头、关系、尾、取值、支撑数
    lines += [_md_table(["头实体", "关系", "尾实体", "取值", "支撑块数"], rows), ""]  # 关系表
    _write(os.path.join(config.EVAL_DIR, "知识图谱统计.md"), "\n".join(lines))  # 写入文档
    return os.path.join(config.EVAL_DIR, "知识图谱统计.md")       # 返回路径


# ---------------------------------------------------------------------------
# 二、检索结果对比与 RAGAS 指标对比
# ---------------------------------------------------------------------------
def _retrieved_cell(item):                                      # 定义检索结果单元格渲染函数
    """把一条检索结果渲染为「第N名 来源第P页(路径)」形式的文本。"""
    via = "·%s" % item["via"] if item.get("via") else ""         # LightRAG 的召回路径标记
    return "第%d名 %s 第%d页%s" % (item["rank"], item["source"], item["page"], via)  # 渲染单元格


def export_retrieval_compare(result):                           # 定义检索结果对比文档函数
    """逐题对比 RAG 与 LightRAG 的检索结果、答案与指标。"""
    rag = result["systems"]["RAG"]["per_question"]               # RAG 逐题结果
    light = result["systems"]["LightRAG"]["per_question"]        # LightRAG 逐题结果
    ref_map = {r["id"]: r for r in result["references"]}         # 参考答案映射
    lines = ["# RAG 与 LightRAG 检索结果对比", "",            # 文档标题
             "每题最多展示前 3 条检索结果与完整答案，`局部`/`全局`/`BM25` 表示该块由哪一路召回。", ""]  # 说明
    rows = []                                                    # 总览表行
    for a, b in zip(rag, light):                                 # 按同一问题配对
        rows.append([a["id"], a["question"][:26] + "…",          # 问题编号与简称
                     "%.3fs" % a["retrieve_seconds"], "%.3fs" % b["retrieve_seconds"],  # 检索耗时
                     a["metrics"]["context_precision"], b["metrics"]["context_precision"]])  # 精度对比
    lines += ["## 一、总览", "", _md_table(                              # 总览小节
        ["id", "问题", "RAG 耗时", "LightRAG 耗时", "RAG 精度", "LightRAG 精度"], rows), ""]  # 总览表
    for a, b in zip(rag, light):                                 # 逐题详细对比
        ref = ref_map.get(a["id"], {})                           # 取该题的参考答案记录
        lines.append("## id=%d %s" % (a["id"], a["question"]))    # 题目标题
        lines.append("")                                         # 空行
        lines.append("**参考答案（银标，%s）**" % ref.get("source", ""))  # 参考答案小节
        lines.append("")                                         # 空行
        lines.append(ref.get("reference", "（无）"))               # 参考答案正文
        lines.append("")                                         # 空行
        for sysname, item in (("RAG", a), ("LightRAG", b)):      # 依次展示两套系统
            lines.append("**%s 检索结果（%.3s 秒，%d 块）**" % (sysname, item["retrieve_seconds"],  # 系统小节标题
                                                                item["retrieved_count"]))  # 检索块数
            lines.append("")                                     # 空行
            for hit in item["retrieved"]:                        # 遍历前 3 条检索结果
                lines.append("- %s：%s" % (_retrieved_cell(hit), hit["preview"]))  # 渲染结果行
            lines.append("")                                     # 空行
            lines.append("**%s 答案**" % sysname)                 # 答案小节
            lines.append("")                                     # 空行
            lines.append(item["answer"])                         # 答案正文
            lines.append("")                                     # 空行
            m = item["metrics"]                                  # 该题指标
            lines.append("**%s 指标**：精度 %.4f / 召回 %.4f / 忠实度 %.4f / 相关性 %.4f"  # 指标行
                         % (sysname, m["context_precision"], m["context_recall"],  # 精度与召回
                            m["faithfulness"], m["answer_relevancy"]))  # 忠实度与相关性
            lines.append("")                                     # 空行
        lines.append("---")                                      # 分题分隔线
        lines.append("")                                         # 空行
    path = os.path.join(config.EVAL_DIR, "检索结果对比.md")       # 输出路径
    _write(path, "\n".join(lines))                               # 写入文档
    return path                                                  # 返回路径


def export_ragas_compare(result):                               # 定义 RAGAS 指标对比文档函数
    """生成两套系统的 RAGAS 指标对比文档（总表 + 逐题明细 + 耗时）。"""
    rag, light = result["systems"]["RAG"], result["systems"]["LightRAG"]  # 取两套系统结果
    lines = ["# RAGAS 评估指标对比", "",                            # 文档标题
             "评估框架：自实现 RAGAS 风格四指标，由 LLM 逐条判定（工单允许「RAGAS 或其他评估框架」）。", "",
             "- 上下文精度 context_precision：有用上下文是否排在前面（越接近 1 越好）",  # 指标说明
             "- 上下文召回 context_recall：参考答案要点被检索上下文覆盖的比例（越接近 1 越好）",  # 指标说明
             "- 忠实度 faithfulness：答案论断能被检索上下文支持的比例（衡量幻觉）",  # 指标说明
             "- 答案相关性 answer_relevancy：答案是否直接回应问题", ""]  # 指标说明
    rows = []                                                   # 总表行
    for r in result["compare"]["rows"]:                         # 遍历对比行
        rows.append([r["metric"], "%.4f" % r["rag"], "%.4f" % r["lightrag"],  # 指标名与两侧取值
                     "%+.4f" % r["delta"], "LightRAG 更优" if r["better"] else "RAG 更优"])  # 变化与结论
    lines += ["## 一、总表（%d 题平均）" % result["meta"]["question_total"], "",  # 总表小节
              _md_table(["指标", "RAG", "LightRAG", "变化", "结论"], rows), "",  # 指标总表
              "LightRAG 更优指标数：**%d/%d**" % (result["compare"]["improved"],
                                                  result["compare"]["total"]), ""]  # 改善汇总
    lines += ["## 二、检索耗时对比", "",                         # 耗时小节标题
              "> 工单十二未规定检索时限，此处仅记录基线耗时；LightRAG 需额外调用一次 LLM 做双层关键词抽取，",  # 口径说明
              "> 耗时高于扁平 RAG 属预期，该耗时将在后续性能优化工单中处理。", ""]  # 口径说明续行
    trows = []                                                  # 耗时表行
    for name in ("RAG", "LightRAG"):                            # 遍历两套系统
        t = result["systems"][name]["timing"]                   # 取耗时统计
        trows.append([name, "%.3f" % t["avg"], "%.3f" % t["max"],  # 系统、平均、最大
                      "%.3f" % t["p95"], "%d/%d" % (t["within_budget"], t["total"])])  # P95 与达标数
    lines += [_md_table(["系统", "平均(s)", "最大(s)", "P95(s)", "≤3s 次数（参考）"], trows), ""]  # 耗时表
    lines += ["## 三、逐题指标明细", ""]                          # 逐题小节
    drows = []                                                  # 逐题表行
    for a, b in zip(rag["per_question"], light["per_question"]):  # 按同一问题配对
        drows.append([a["id"], a["question"][:20] + "…",         # 问题编号与简称
                      "%.2f" % a["metrics"]["context_precision"], "%.2f" % b["metrics"]["context_precision"],  # 精度
                      "%.2f" % a["metrics"]["context_recall"], "%.2f" % b["metrics"]["context_recall"],  # 召回
                      "%.2f" % a["metrics"]["faithfulness"], "%.2f" % b["metrics"]["faithfulness"],  # 忠实度
                      "%.2f" % a["metrics"]["answer_relevancy"], "%.2f" % b["metrics"]["answer_relevancy"]])  # 相关性
    lines += [_md_table(["id", "问题", "RAG精度", "LR精度", "RAG召回", "LR召回",  # 表头
                         "RAG忠实", "LR忠实", "RAG相关", "LR相关"], drows), ""]  # 逐题明细表
    path = os.path.join(config.EVAL_DIR, "RAGAS指标对比.md")      # 输出路径
    _write(path, "\n".join(lines))                               # 写入文档
    return path                                                  # 返回路径


def export_all(result, graph):                                  # 定义交付物生成函数
    """生成全部评估交付物：图谱统计、图谱图、检索对比、指标对比。"""
    export_graph_report(graph)                                  # 生成图谱统计文档
    draw_graph(graph)                                           # 生成图谱可视化
    export_retrieval_compare(result)                            # 生成检索结果对比
    export_ragas_compare(result)                                # 生成 RAGAS 指标对比
    LOGGER.info("全部评估交付物已生成于 %s", config.EVAL_DIR)      # 输出完成日志


if __name__ == "__main__":                                      # 仅在直接运行时执行
    with open(config.EVAL_RESULT_FILE, "r", encoding="utf-8") as f:  # 打开评估结果
        _result = json.load(f)                                  # 读取评估结果
    with open(config.GRAPH_FILE, "r", encoding="utf-8") as f:   # 打开知识图谱
        _graph = json.load(f)                                   # 读取图谱
    export_all(_result, _graph)                                 # 生成全部交付物
