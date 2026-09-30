"""问答链路人工冒烟。

存在的理由：自动化测试能证明"引用没编造"，但证明不了"答案读起来对不对"——
3b 模型完全可能给出一条引用全对、结论却答非所问的回复。这是 ③a 唯一无法
自动化的验收动作（②期冒烟集也有同样的性质），所以必须有个趁手的入口。

用法：
    cd D:/xinzg6/fl && python tools/ask.py "租房押金不退怎么办"
    cd D:/xinzg6/fl && python tools/ask.py -s public "房东不退押金"
    cd D:/xinzg6/fl && python tools/ask.py -s public --no-extras "房东不退押金"
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

SIDES = ("internal", "public")


def build_parser() -> argparse.ArgumentParser:
    """参数解析独立成函数，便于单测不真跑模型。"""
    parser = argparse.ArgumentParser(description="法律 RAG 问答冒烟")
    parser.add_argument("question", help="要问的问题")
    parser.add_argument("-s", "--side", choices=SIDES, default="internal",
                        help="internal=律师侧（本地 Ollama）；public=公众侧（DeepSeek）")
    # 默认开：附加区块就是 ③b-1 的交付本身。留这个开关是为了能在**同一条链路**
    # 上跑出「没有区块」的对照输出（③a 原路径），而不是靠改代码来对照
    parser.add_argument("--no-extras", action="store_true",
                        help="不构造附加区块（案由/示例律师/费用），走 ③a 原路径")
    return parser


def _citation_label(cite) -> str:
    """渲染一条引用的条/款/项标签。

    article 的契约允许两种写法（schema.py：「第五百八十四条」或「584」），
    已带「第…条」的照原样用、否则才补外壳——不加这一步中文条号会被包两遍，
    渲染成「第第五百八十四条条」（Task 13 冒烟 M1 的原始标本）。款/项为空时
    整段省略，不留「款 项」两个空占位。
    """
    raw = (cite.article or "").strip()
    # 分两步补外壳：先保证有「第」，再保证有「条」。「第584」这种漏了尾字的
    # 半截写法若按「首尾同时匹配才原样用」处理，会被包成「第第584条」
    label = raw if raw.startswith("第") else f"第{raw}"
    if not label.endswith("条"):
        label += "条"
    if cite.paragraph:
        label += f"第{cite.paragraph}款"
    if cite.item:
        label += f"第{cite.item}项"
    return label


def format_result(result) -> str:
    """把 QAResult 渲染成人看的文本。"""
    lines = [f"状态：{result.status}    生成次数：{result.attempts}", "",
             result.answer]
    if result.disclaimer:
        lines += ["", f"【免责】{result.disclaimer}"]
    if result.failures:
        lines += ["", "【未通过原因】"] + [f"  - {f}" for f in result.failures]
    if result.citations:
        lines += ["", "【引用】"]
        for cite in result.citations:
            lines.append(f"  - {_citation_label(cite)}：{cite.quote[:40]}…")
    if result.sources:
        lines += ["", "【检索到的原文】"]
        for block in result.sources:
            score = "" if block.get("rerank_score") is None else f"  精排 {block['rerank_score']:.3f}"
            lines.append(f"  - 第{block['article_no']}条（{block['path']}）"
                         f"[{block['source']}]{score}")
    if result.extras:
        # 有没有区块由 QAResult 决定（律师侧恒无；公众侧只有 out_of_scope 里**不含
        # 问价词**的那类出口无 —— 判据收在 Answerer._extras_for 一处），渲染层不再
        # 重复判断一次侧别 / 状态 / 问价词，免得两处口径分叉
        lines.append(format_extras(result.extras))
    return "\n".join(lines)


def format_extras(block: dict | None) -> str:
    """渲染附加区块。没给区块（律师侧 / 无区块的出口）就返回空串，调用方直接拼接。

    区块的字段形状由 `attach.build` 定（顶层恰为 cause/field/lawyers/fee/
    disclaimer，**fee 恒 9 键**：含 `charge_basis`），这里按契约取用。
    """
    if not block:
        return ""
    lines = [f"\n【专业领域】{block['field']}"]
    if block.get("cause"):
        # 案由认不出时 tag 为 None：那不是故障，省掉这行而不是印「None」
        lines.append(f"【案由】{block['cause']}")
    if block["lawyers"]:
        lines.append("【示例律师】（演示数据）")
        for card in block["lawyers"]:
            # note 用 get：DEMO_MODE 关掉后卡片只有 5 键（无 note），
            # card['note'] 会当场 KeyError，而那正是「已接真实名录」的状态
            note = f"（{card['note']}）" if card.get("note") else ""
            lines.append(f"  - {card['name']}{note}｜{card['org']}｜"
                         f"{card['field']}｜{card['contact_hint']}")
    fee = block.get("fee") or {}
    if fee.get("status") == "ok":
        # unit 缺省（None/空串）不印。有 unit 必须紧跟数字印出来：语料写
        # 「1万元」时只印「1 ~ 10」是静默错 4 个数量级（Task 4 复审点名的洞）
        unit = fee.get("unit") or ""
        # 计价基础附在费用行上（用户 2026-09-29 裁决，满足 FR-9.4「含计费方式说明」）：
        # 不读依据原文也能看懂计价方式 —— 真跑实证 N10 就是「/有效工作小时」被丢掉后
        # 「1000 ~ 8000元」被读成一次性收费。分隔符「/」由渲染层补（模型照抄的是词
        # 本身），若模型把片段里的「/有效工作小时」连斜杠一起抄回来则去掉，免得印「//」
        charge = (fee.get("charge_basis") or "").lstrip("/")
        charge = f"/{charge}" if charge else ""
        # 依据字段两侧都可能为空（fees.py 用 snippet.get 取值，不保证非空），
        # 照原样印会渲染出「（依据：None None）」。两边都空就整段省略；只有一边
        # 时不省略——单靠文档名也比什么都不给强，且空位不会变成字面的 None
        basis = " ".join(p for p in (fee.get("source_doc"), fee.get("source_no")) if p)
        basis = f"（依据：{basis}）" if basis else ""
        lines.append(f"【费用区间】{fee['low']} ~ {fee['high']}{unit}{charge}{basis}")
        # 片段原文照印（用户 2026-09-29 裁决）：真链路实测两处误导都只有原文能拆穿 ——
        # 「1000 ~ 8000元」丢掉了「/有效工作小时」，风险代理的「5 ~ 15%」是回款提成
        # 而适用前提在片段里。回查只保证数字有字面出处（fees.py 模块 docstring 已声明
        # 它挡不住用错语境），所以用户要能看见并核对原文，而不是只能信「AI 说」。
        # basis 是 estimate 的 ok 支给的命中片段（设计允许为空），空则整段不印
        if fee.get("basis"):
            # 片段最长 821 字且含换行（语料按语义单元分段），直接拼进一行会把上面几行
            # 的格式冲散；逐行缩进两格印，多行片段读下来仍是「一整段依据」。这里**有意**
            # 不截断：本 CLI 的用途就是人工核对全文，④ 期公众界面是否节选另有设计
            lines.append("【依据原文】")
            lines.extend(f"  {line}" for line in fee["basis"].splitlines())
    elif fee.get("status") == "no_corpus":
        lines.append("【费用区间】暂无费用口径依据")
    elif fee.get("status") == "rejected":
        # 判拒有两种来由，给用户的意思不同：设计 §七 要求「命中片段非现行 → 不采用」
        # 这一支**在报告里说明**，一律印「无法指回收费口径」会把「口径过期、我们没
        # 采用它」错说成「模型编了数」。按 estimate 写的 reason 关键词分流；**不印
        # reason 原文**——那是内部措辞，且与 unavailable 不印异常同一个安全理由
        # （面向公众的文案在渲染层自己定，不从返回值透传）
        if "效力" in (fee.get("reason") or ""):
            lines.append("【费用区间】命中的收费口径效力非现行有效，未采用该口径")
        else:
            lines.append("【费用区间】生成结果无法指回收费口径，已按不猜数处理")
    elif fee.get("status") == "unavailable":
        # 只给通用文案：`reason` 里带的是原始异常（如 milvus down），那是给
        # 开发者排查的现场，印给公众既是噪音也泄露内部实现
        lines.append("【费用区间】费用信息暂时不可用")
    lines.append(f"【提示】{block['disclaimer']}")
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    # 依赖在函数内导入：单测导入本模块时不触发模型加载
    from app.core.factory import build_services
    from app.generation.profiles import SIDE_PUBLIC

    # 装配（连库、加载两个模型、按侧决定要不要附加区块）整段搬进了共享工厂，
    # 与 HTTP 服务用的是同一份 —— 本文件只剩下参数、渲染与打印。
    # 律师侧从不构造附加区块（设计红线：数据不出域），副产物是它不再被
    # 「DeepSeek 密钥缺失」牵连，而那本不是它要用的东西；工厂还把这个方向
    # 钉成了硬拦截（律师侧 + 要区块 = 调用方的编程错误）
    services = build_services(args.side,
                              with_extras=args.side == SIDE_PUBLIC
                              and not args.no_extras)
    print(format_result(services.answerer.answer(args.question, args.side)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
