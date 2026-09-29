# -*- coding: utf-8 -*-
"""把系统结构画成思维导图（PNG + SVG）。

    .venv/bin/python -m scripts.draw_mindmap

WSL 里通常没有中文字体，这里直接借用 Windows 的微软雅黑；
若在其他环境运行，可通过 --font 指定字体文件。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib import font_manager                                # noqa: E402
from matplotlib.patches import FancyBboxPatch                      # noqa: E402

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")

# 画布配色（与文档风格一致，深浅分明）
C_BG = "#ffffff"
C_ROOT = "#1f4e79"
C_ROOT_TXT = "#ffffff"
BRANCH_COLORS = [
    ("#2e75b6", "#deeaf6"), ("#548235", "#e2efda"), ("#bf8f00", "#fff2cc"),
    ("#c55a11", "#fbe5d6"), ("#7030a0", "#e9e0f5"), ("#0f7b8a", "#d9eef1"),
    ("#a03030", "#f8dede"), ("#4a5568", "#e8eaee"), ("#7a5195", "#ece3f2"),
]

MINDMAP = [
    ("对话", [
        "问答接口 /api/chat/ask",
        "流式接口 /api/chat/stream",
        "多轮会话（session_id 延续）",
        "追问改写：那它呢 → 完整问题",
        "查询扩写：1 问 → 3 个检索查询",
    ]),
    ("检索", [
        "稠密向量：BGE-M3（1024 维）",
        "稀疏权重：BM25 风格，命中关键词",
        "元数据路：按法条号精确命中",
        "RRF 加权融合（元数据路 2.5 倍权）",
        "BGE-reranker-v2-m3 交叉编码器精排",
        "降级路径：混合检索失败退纯向量",
    ]),
    ("文档接入", [
        "MinerU 版面 / OCR / 表格一体",
        "独立 venv 子进程，basic 档",
        "去水印：文字 / 图片 / 元数据",
        "降级三件套：PyMuPDF 文本层",
        "+ PaddleOCR 图表文字",
        "+ pdfplumber 还原表格",
        "use_ocr / use_tables 只作用于降级",
        "语义分块：按话题转折切分",
        "实测：28 页报告 15174 字 / 38 秒",
    ]),
    ("记忆", [
        "短期：Redis List，最近 10 轮原文",
        "长期：对话摘要 + 向量存 Milvus",
        "溢出缓冲 mem:pending，攒 6 行再摘要",
        "按 user_id + role_id 双重隔离",
    ]),
    ("知识库", [
        "rag_knowledge：13364 个切片",
        "三个角色均 4000 条以上",
        "法条 4555 · 心理 4675 · 金融 4134",
        "法条覆盖 21 部法规",
        "WHO 报告 / Fin-Eva 金融数据",
        "MySQL law_index 法条索引表",
    ]),
    ("角色", [
        "律师 / 心理医生 / 金融理财师",
        "人设与规则存 MySQL，可扩展",
        "兜底话术 · 免责声明自动追加",
        "会话归属校验（越权返回 403）",
    ]),
    ("接口", [
        "问答 chat · 角色 role",
        "知识 knowledge · 系统 system",
        "统一 Resp 包装 code/msg/data",
        "OpenAPI 文档 /docs",
    ]),
    ("框架", [
        "LangChain 六大功能",
        "Models：自定义 BaseChatModel",
        "Prompts：ChatPromptTemplate",
        "Chains：LCEL 组链",
        "Memory：Redis 消息历史接口",
        "Indexes / Agents：保留自研",
    ]),
    ("质量与治理", [
        "引用校验：法条号防幻觉",
        "知识库去重：完全重复 + 近似重复",
        "RAGAS 四指标 + 56 题评测集",
        "pytest 49 项 / newman 20 请求",
        "JMeter 压测（读接口 1600 次）",
        "边界排查：15 个场景",
    ]),
    ("部署", [
        "install.sh 环境检查与安装",
        "run.sh 启动 · shutdown.sh 停止",
        "docker-compose 拉起依赖服务",
        "Ubuntu / 算力云 / 腾讯云 / 阿里云",
    ]),
    ("存储", [
        "MySQL：用户 / 角色 / 会话 / 消息 / 法条索引",
        "Milvus：向量 + 稀疏 + 元数据字段",
        "Redis：List / Hash / Set / zSet / String",
        "日志：标准库 logging 三路输出",
    ]),
]


def setup_font(font_path=None):
    """优先用参数指定的字体，其次借用 Windows 中文字体。"""
    candidates = [font_path] if font_path else []
    candidates += [
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            font_manager.fontManager.addfont(p)
            name = font_manager.FontProperties(fname=p).get_name()
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            print("使用字体: %s (%s)" % (name, p))
            return True
    print("警告: 未找到中文字体，图中中文可能显示为方块")
    return False


# 布局常量（单位是数据坐标）
CHILD_GAP = 0.36        # 子项行距
BRANCH_PAD = 0.85       # 分支标题占的高度
BRANCH_GAP = 0.55       # 分支之间的间隔
X_CENTER = 2.0          # 根节点到分支的水平距离
X_BRANCH = 3.1          # 分支标题中心
XLIM = 10.8             # 画布半宽


def draw(out_png, out_svg=None, font_path=None):
    setup_font(font_path)

    left = MINDMAP[:5]
    right = MINDMAP[5:]

    def layout(items):
        """算出每个分支的中心 y，以及整列的总高度。"""
        heights = [len(ch) * CHILD_GAP + BRANCH_PAD for _, ch in items]
        total = sum(heights) + BRANCH_GAP * (len(items) - 1)
        out, y = [], total / 2
        for h in heights:
            out.append(y - h / 2)
            y -= h + BRANCH_GAP
        return out, total

    left_y, left_total = layout(left)
    right_y, right_total = layout(right)

    # 画布高度按内容实际高度决定，避免最高的那列被裁掉
    half = max(left_total, right_total) / 2 + 0.7
    fig_w = 22.0
    fig_h = fig_w * (2 * half) / (2 * XLIM)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-XLIM, XLIM)
    ax.set_ylim(-half, half)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)

    # 根节点
    ax.add_patch(FancyBboxPatch((-2.05, -0.62), 4.1, 1.24,
                                boxstyle="round,pad=0.12,rounding_size=0.25",
                                linewidth=0, facecolor=C_ROOT, zorder=3))
    ax.text(0, 0.04, "基于 RAG 的\n角色扮演系统",
            ha="center", va="center", fontsize=22, color=C_ROOT_TXT,
            fontweight="bold", zorder=4, linespacing=1.5)

    def draw_side(items, centers, direction, color_offset):
        """direction = -1 左侧，+1 右侧"""
        bx = direction * X_BRANCH
        for idx, ((name, children), y_center) in enumerate(zip(items, centers)):
            color = BRANCH_COLORS[(idx + color_offset) % len(BRANCH_COLORS)][0]

            # 根 -> 分支 的连接线（简单贝塞尔，比折线干净）
            x_start = direction * X_CENTER
            x_end = bx - direction * 1.28
            mid = (x_start + x_end) / 2
            ax.plot([x_start, mid, mid, x_end], [0, 0, y_center, y_center],
                    color=color, linewidth=2.0, alpha=0.42,
                    solid_capstyle="round", zorder=1)

            # 分支标题
            ax.add_patch(FancyBboxPatch(
                (bx - 1.22, y_center - 0.33), 2.44, 0.66,
                boxstyle="round,pad=0.08,rounding_size=0.16",
                linewidth=0, facecolor=color, zorder=3))
            ax.text(bx, y_center, name, ha="center", va="center",
                    fontsize=14, color="#ffffff", fontweight="bold", zorder=4)

            # 子项：竖线串起来，每个点一条小横线
            if not children:
                continue
            x_line = bx + direction * 1.40
            top_y = y_center + CHILD_GAP * 0.5
            bot_y = top_y - (len(children) - 1) * CHILD_GAP
            ax.plot([x_line, x_line], [top_y, bot_y], color=color,
                    linewidth=1.1, alpha=0.45, zorder=1)
            for j, child in enumerate(children):
                cy = top_y - j * CHILD_GAP
                ax.plot([x_line, x_line + direction * 0.22], [cy, cy],
                        color=color, linewidth=1.1, alpha=0.45, zorder=1)
                ax.text(x_line + direction * 0.34, cy, child,
                        ha="left" if direction > 0 else "right",
                        va="center", fontsize=11, color="#333333", zorder=4)

    draw_side(left, left_y, -1, 0)
    draw_side(right, right_y, 1, 5)

    fig.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor=C_BG, pad_inches=0.3)
    if out_svg:
        fig.savefig(out_svg, bbox_inches="tight",
                    facecolor=C_BG, pad_inches=0.3)
    plt.close(fig)

    print("已生成: %s" % out_png)
    if out_svg:
        print("已生成: %s" % out_svg)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DOCS_DIR, "思维导图.png"))
    ap.add_argument("--font", default=None)
    args = ap.parse_args()

    svg = os.path.splitext(args.out)[0] + ".svg"
    draw(args.out, svg, args.font)


if __name__ == "__main__":
    main()
