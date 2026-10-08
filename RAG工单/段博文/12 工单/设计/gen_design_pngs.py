# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-LightRAG优化
"""生成工单12设计 PNG（5 张）。"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent
C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE, C_GRAY = (
    "#2f6fed", "#27ae60", "#e67e22", "#e74c3c", "#8e44ad", "#95a5a6")


def box(ax, x, y, w, h, text, fc, fontsize=10.5, tc="white", lw=0):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012",
        fc=fc, ec="none", lw=lw, mutation_aspect=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=tc, fontweight="bold", linespacing=1.5)


def arrow(ax, x1, y1, x2, y2, color="#555", lw=1.8, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def canvas(figsize=(14, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"已生成 {name}")


# ============ 01 LightRAG 总体架构图 ============
fig, ax = canvas((14.5, 8.2))
ax.text(50, 96, "LightRAG 总体架构（索引构建 + 双层检索）", ha="center",
        fontsize=17, fontweight="bold")

box(ax, 3, 82, 20, 8, "招股说明书1.pdf（兴图新科）\n招股说明书2.pdf（力源信息）", C_GRAY)
box(ax, 27, 82, 16, 8, "PDF 解析\n文本清洗", C_BLUE)
box(ax, 47, 82, 16, 8, "文本切块\nChunk (1024 token)", C_BLUE)
box(ax, 67, 82, 30, 8, "LLM 实体关系抽取（DeepSeek）\n定制 13 类实体 / 13 类关系", C_ORANGE)

box(ax, 55, 62, 20, 10, "知识图谱存储\n实体节点 + 关系边\n（NetworkX 图存储）", C_GREEN)
box(ax, 79, 62, 18, 10, "向量存储\n实体/关系/文本块向量\n（bge-m3 · 1024维）", C_GREEN)

arrow(ax, 23, 86, 27, 86); arrow(ax, 43, 86, 47, 86); arrow(ax, 63, 86, 67, 86)
arrow(ax, 82, 82, 65, 72); arrow(ax, 90, 82, 90, 72)

box(ax, 3, 62, 24, 10, "增量更新\n图结构差异分析\n只更新相关部分", C_ORANGE)
arrow(ax, 15, 72, 55, 69, color=C_ORANGE, style="<|-|>")

box(ax, 8, 38, 38, 12, "查询层（双层检索机制）\n低层检索：具体实体 → 实体向量 + 邻域扩展\n高层检索：广泛主题 → 关系/主题向量", C_RED)
arrow(ax, 60, 62, 40, 50); arrow(ax, 84, 62, 44, 47)

box(ax, 50, 38, 20, 12, "图扩展\n相关块回溯\n上下文融合去重", C_PURPLE)
box(ax, 74, 38, 22, 12, "上下文组装\n→ LLM 生成答案\n（DeepSeek）", C_BLUE)
arrow(ax, 46, 44, 50, 44); arrow(ax, 70, 44, 74, 44)

box(ax, 20, 18, 60, 8, "对比评估：传统 RAG（扁平向量检索） vs LightRAG（图谱双层检索）\nRAGAS 指标：context_precision / context_recall / faithfulness / answer_relevancy",
    "#1a1a2e")
arrow(ax, 27, 38, 40, 26); arrow(ax, 74, 38, 62, 26)
save(fig, "01-LightRAG总体架构图.png")

# ============ 02 知识图谱构建流程图 ============
fig, ax = canvas((14.5, 7.6))
ax.text(50, 95, "知识图谱构建流程（含增量更新）", ha="center", fontsize=17, fontweight="bold")

steps = [
    (2, "PDF 解析\npymupdf 逐页提取\n清洗页眉页脚", C_GRAY),
    (21, "全文拼接\n兴图新科 44.8万字\n力源信息 30.9万字", C_GRAY),
    (40, "文本切块\n1024 token / 块\n100 token 重叠", C_BLUE),
    (59, "实体抽取（LLM）\n公司/人员/股东/产品\n行业/财务指标/募投项目\n技术标准/工程奖项…13类", C_ORANGE),
    (59, ""),
]
box(ax, 2, 58, 17, 22, "PDF 解析\npymupdf 逐页提取\n清洗页眉页脚", C_GRAY)
box(ax, 21, 58, 17, 22, "全文拼接\n兴图新科 44.8万字\n力源信息 30.9万字", C_GRAY)
box(ax, 40, 58, 17, 22, "文本切块\n1024 token / 块\n100 token 重叠", C_BLUE)
box(ax, 59, 55, 20, 25, "实体抽取（LLM）\n公司/人员/股东/产品\n行业/财务指标/募投项目\n技术标准/工程奖项…13类", C_ORANGE)
box(ax, 81, 55, 17, 25, "关系抽取（LLM）\n控股/持股 任职\n供应 客户 所属行业\n募投投资 参与制定\n荣获奖项 拥有资质…13类", C_ORANGE)

arrow(ax, 19, 69, 21, 69); arrow(ax, 38, 69, 40, 69); arrow(ax, 57, 69, 59, 69)
arrow(ax, 79, 69, 81, 69)

box(ax, 40, 28, 17, 18, "去重与合并\n同名实体合并\n描述摘要融合\n(DeepSeek 总结)", C_GREEN)
box(ax, 59, 28, 20, 18, "图谱持久化\n实体节点 + 关系边\n(NetworkX/graphml)\n向量入库 (bge-m3)", C_GREEN)
box(ax, 81, 28, 17, 18, "KV 存储\n文本块 / 文档状态\n(LLM 结果缓存)", C_GREEN)
arrow(ax, 66, 55, 50, 46); arrow(ax, 90, 55, 69, 46); arrow(ax, 57, 37, 59, 37)
arrow(ax, 79, 37, 81, 37)

box(ax, 2, 28, 34, 18, "★ 增量更新（LightRAG 优势）\n插入招股书2时：对新 chunk 抽取实体 → 与已有图谱\n做差异分析 → 只合并新实体/新边 → 无需全库重建\n实测：文档1 索引后，增量插入文档2 直接追加", C_RED)
arrow(ax, 40, 55, 20, 46)

box(ax, 2, 6, 96, 14, "图谱产出：兴图新科 + 力源信息 招股书知识图谱\n实体类型：公司(Company)、人员(Person)、股东(Shareholder)、产品与服务(Product)、行业(Industry)、财务指标(FinancialMetric)、\n募投项目(Project)、技术标准(TechnicalStandard)、工程奖项(EngineeringAward)、技术与资质(TechnologyQualification)、机构(Institution)、地点(Location)、事件(Event)",
    "#1a1a2e", fontsize=9.5)
save(fig, "02-知识图谱构建流程图.png")

# ============ 03 双层检索机制图 ============
fig, ax = canvas((14, 8))
ax.text(50, 95, "LightRAG 双层检索机制（hybrid 模式）", ha="center", fontsize=17, fontweight="bold")

box(ax, 3, 78, 24, 10, "用户问题\n“兴图新科参与制定了哪个技术标准？”", C_GRAY)
box(ax, 33, 78, 20, 10, "LLM 关键词提取\n（DeepSeek）", C_ORANGE)
arrow(ax, 27, 83, 33, 83)

box(ax, 8, 56, 34, 14, "低层检索（Local / 具体实体）\n局部关键词：兴图新科、视频指挥系统技术标准\n→ 实体向量匹配 → 选中节点 → 图邻域扩展\n→ 关联 chunk 回溯", C_RED)
box(ax, 56, 56, 34, 14, "高层检索（Global / 主题概念）\n全局关键词：技术标准、军民融合、视频指挥\n→ 关系/主题向量匹配（关系描述）\n→ 关联实体与 chunk 回溯", C_PURPLE)
arrow(ax, 43, 80, 25, 70); arrow(ax, 48, 80, 73, 70)

box(ax, 25, 34, 48, 12, "双层结果融合\n实体 + 关系 + 源文本 chunk 去重排序\n按 token 预算组装上下文（Entities / Relationships / Sources）", C_GREEN)
arrow(ax, 25, 58, 40, 46); arrow(ax, 73, 58, 58, 46)

box(ax, 33, 14, 32, 10, "→ LLM 生成最终答案（DeepSeek）", C_BLUE)
arrow(ax, 49, 34, 49, 24)

box(ax, 3, 6, 92, 5.2, "对比传统 RAG：仅查询向量 vs chunk 向量匹配（扁平检索），无实体/关系结构，多实体关联与主题归纳类问题检索断层",
    "#fdebd0", tc="#a04000", fontsize=10)
save(fig, "03-双层检索机制图.png")

# ============ 04 实体关系类型设计图 ============
fig, ax = canvas((14.5, 9))
ax.text(50, 97, "招股书领域实体类型与关系类型设计（注入 entity_types_guidance）",
        ha="center", fontsize=16.5, fontweight="bold")

ents = [
    ("公司(Company)", "发行人/子公司/关联方/供应商/客户/券商/会所"),
    ("人员(Person)", "法定代表人/董监高/核心技术人员"),
    ("股东(Shareholder)", "控股股东/发起人/机构投资者"),
    ("产品与服务(Product)", "电子元器件分销/专用通信设备"),
    ("行业(Industry)", "所处行业/上下游细分行业"),
    ("财务指标(FinancialMetric)", "收入/净利/毛利率(注明报告期与金额)"),
    ("募投项目(Project)", "募集资金投资项目/补充流动资金"),
    ("技术标准(TechnicalStandard)", "国标/行标/军用标准"),
    ("工程奖项(EngineeringAward)", "国家科技进步一等奖工程"),
    ("技术与资质(TechQualification)", "专利/软著/军工资质/认证"),
    ("机构(Institution)", "证监会/军队单位/行业协会"),
    ("地点(Location)", "注册地/生产基地/募投实施地"),
    ("事件(Event)", "IPO/重大合同/股权变动"),
]
rels = [
    ("控股/持股", "股东 → 公司（含持股比例）"),
    ("任职", "人员 → 公司/股东（含职务）"),
    ("供应关系", "供应商 → 公司"),
    ("客户关系", "客户 → 公司"),
    ("子公司关系", "子公司 → 母公司"),
    ("所属行业", "公司/产品 → 行业（上/下游）"),
    ("募投投资", "公司 → 募投项目（含金额）"),
    ("参与制定", "公司 → 技术标准"),
    ("荣获奖项", "公司 → 工程奖项"),
    ("拥有资质", "公司 → 技术与资质"),
    ("位于", "公司/项目 → 地点"),
    ("竞争关系", "公司 → 公司"),
    ("中介服务", "保荐/审计/律所 → 发行人"),
]

ax.text(28, 90, "13 类实体类型", ha="center", fontsize=13, fontweight="bold", color=C_BLUE)
for i, (name, desc) in enumerate(ents):
    y = 84 - i * 6.4
    box(ax, 2, y, 52, 5.6, f"{name}　—　{desc}",
        "#eaf2fd", tc="#1a1a2e", fontsize=9.5)

ax.text(77, 90, "13 类关系类型", ha="center", fontsize=13, fontweight="bold", color=C_ORANGE)
for i, (name, desc) in enumerate(rels):
    y = 84 - i * 6.4
    box(ax, 57, y, 41, 5.6, f"{name}：{desc}",
        "#fdf2e9", tc="#1a1a2e", fontsize=9.5)

box(ax, 2, 0.5, 96, 5, "抽取要求：实体用全称并保持一致；比例/金额/报告期必须入描述；关系描述体现业务实质（如“力源信息是德州仪器中国一级授权分销商”）",
    "#1a1a2e", fontsize=10)
save(fig, "04-实体关系类型设计图.png")

# ============ 05 对比评估方案图 ============
fig, ax = canvas((14.5, 7.8))
ax.text(50, 95, "RAG vs LightRAG 对比评估方案（16 题 · RAGAS 指标）",
        ha="center", fontsize=17, fontweight="bold")

box(ax, 2, 74, 20, 12, "测试问题集\n工单指定 16 题\n(力源 6 题 / 兴图新科 10 题)\n含参考答案与关键词", C_GRAY)

box(ax, 27, 82, 26, 8, "链路A：传统 RAG\nbge-m3 扁平向量检索 Top-8", C_RED)
box(ax, 27, 68, 26, 8, "链路B：LightRAG\nhybrid 双层检索（实体+关系+chunk）", C_GREEN)
arrow(ax, 22, 80, 27, 85); arrow(ax, 22, 77, 27, 72)

box(ax, 58, 82, 18, 8, "上下文\n(同题同块来源)", C_BLUE)
box(ax, 58, 68, 18, 8, "上下文\n(Entities/Relations/Sources)", C_BLUE)
arrow(ax, 53, 86, 58, 86); arrow(ax, 53, 72, 58, 72)

box(ax, 80, 68, 18, 22, "DeepSeek 生成答案\n(temperature=0\n两链路共用同一\n生成器，公平对比)", C_ORANGE)
arrow(ax, 76, 86, 80, 84); arrow(ax, 76, 72, 80, 76)

box(ax, 27, 40, 60, 16, "RAGAS 简化评估（以参考答案关键词为基准）\ncontext_precision　检索上下文中相关内容的排名加权占比\ncontext_recall　参考答案关键词被上下文覆盖比例\nfaithfulness　答案事实可归因于上下文的比例\nanswer_relevancy　答案对参考答案关键词覆盖比例", "#1a1a2e", fontsize=10.5)
arrow(ax, 89, 68, 70, 56)

box(ax, 27, 12, 60, 16, "产出：逐题检索结果对比表（上下文关键词命中 / 答案对比）\nRAGAS 四指标汇总对比 + 平均检索耗时对比\n分析结论：多实体关联 / 主题归纳题 LightRAG 优势显著；\n简单事实题两者相当", C_GREEN)
arrow(ax, 57, 40, 57, 28)
save(fig, "05-对比评估方案图.png")

print("全部设计图生成完成")
