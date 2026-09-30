#!/usr/bin/env python
"""RAGAS 风格评测（MVP 简化版）：用 LLM 做 judge，计算 faithfulness 与 context_relevancy。

依赖：LLM 已配置（Ollama 或在线 API）。生产可替换为完整 ragas 库。
用法（--role 决定用哪组题，两个角色各有一组）：
    python eval/ragas_eval.py --role lawyer

    # 本机实际用法（judge 换模型时也走这个入口）：
    .venv/bin/python eval/ragas_eval.py --role psychologist --judge-model qwen3:8b

前置条件与副作用：
    - Milvus、关系库、Ollama 都要可达；judge 模型必须已 pull 到本地。跑之前先停掉
      web 服务（Milvus Lite 单进程独占，服务开着会 DataDirLockedError）。
    - **只读知识库**，不写入库也不改任何文档；但会经 pipeline.answer 真的跑一轮对话，
      于是在记忆后端留下 eval-0 … eval-N 这些会话（.env 默认 MEMORY_BACKEND=redis 就是
      写进 Redis；Redis 不可达时自动降级为进程内内存，只打一条警告，评测照常出分）。
    - 没有任何断言语义：退出码恒为 0，分数要人看输出。恒值时 _warn_if_constant 会告警，
      但**告警不等于 judge 坏了**——评测集本身没区分度时也会恒值，先怀疑题，再怀疑模型。
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.core.llm import LLMClient  # noqa: E402
from app.core.logging_config import get_logger, setup_logging  # noqa: E402
from app.core.pipeline import RAGPipeline  # noqa: E402

log = get_logger("ragas_eval")

# 评测样例（问题 + 参考答案），**按角色分组**：两个角色的知识库完全不同（律师是法条、
# 心理咨询师是咨询对话 + 精神卫生法），拿一套题跨角色评测没有意义——另一个角色必然零召回，
# 测出来的低分是「题不对库」，不是模型不行。
#
# reference 仅供人工核对，judge 只看 问题/答案/上下文 三项。
# 每组都覆盖「知识库内有」的常规题 + 一条「知识库没有」的陷阱题，后者用来测 BR-1
# （资料没有就明确告知、不得编造），也让 context_relevancy 不至于恒为 1.0 失去区分度。
# 陷阱题的关键词都核实过不在语料里（如「永久居留」在律师语料中 0 命中），否则就不是陷阱。
QA_SAMPLES: dict[str, list[dict]] = {
    "lawyer": [
        {"question": "试用期最长可以约定多久？",
         "reference": "三年以上固定期限和无固定期限的劳动合同，试用期不得超过六个月"},
        {"question": "用人单位超过一个月不满一年未与劳动者订立书面劳动合同，要承担什么后果？",
         "reference": "应当向劳动者每月支付二倍的工资"},
        {"question": "民法典规定的普通诉讼时效期间是多久？", "reference": "三年"},
        {"question": "从建筑物中抛掷物品造成他人损害，物业服务企业要担责吗？",
         "reference": "物业服务企业等建筑物管理人未采取必要的安全保障措施的，"
                      "应当依法承担未履行安全保障义务的侵权责任"},
        {"question": "刑法规定已满多少周岁的人犯罪应当负刑事责任？", "reference": "已满十六周岁"},
        {"question": "外国人在中国申请永久居留需要提交哪些材料？",
         "reference": "知识库无此内容，应明确告知无法回答、不得编造"},
    ],
    "psychologist": [
        {"question": "精神卫生法规定，精神障碍的住院治疗实行什么原则？", "reference": "自愿原则"},
        {"question": "精神障碍的诊断应当由谁作出？", "reference": "由精神科执业医师作出"},
        {"question": "心理咨询人员可以从事心理治疗或者精神障碍的诊断、治疗吗？",
         "reference": "不得从事心理治疗或者精神障碍的诊断、治疗；"
                      "发现接受咨询的人员可能患有精神障碍的，应当建议其到医疗机构就诊"},
        {"question": "精神障碍患者的隐私信息受什么保护？",
         "reference": "有关单位和个人应当对精神障碍患者的姓名、肖像、住址、工作单位、"
                      "病历资料等可能推断出其身份的信息予以保密"},
        {"question": "长期失眠可以怎么调整？",
         "reference": "语料是咨询对话集，无固定标准答案；judge 看的是「有没有依据检索到的"
                      "对话内容作答、有没有编造」"},
        {"question": "心理咨询师资格证怎么报考？",
         "reference": "知识库无此内容，应明确告知无法回答、不得编造"},
    ],
}


# 只用来从 judge 正文里抠出**第一个**数字，所以不锚定整串：弱模型常写成
# "0.7"、"评分：0.7"、"0.7 分"，甚至有先写解释再给分的；这些都要能取到。
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+")


def _ask_score(llm: LLMClient, prompt: str) -> tuple[float, str]:
    """返回 (分数, judge 的理由)。

    理由一并带出来是为了可审计：分数只是结论，判断依据在理由里——「资料只有一句
    结论、答案却扩写了一整篇」这类问题，只看分数无法区分「judge 误判」和「模型真
    的在编」，必须看它凭什么这么判。

    理由优先取推理段（qwen3 等推理模型的思维链放在独立字段里），没有推理段才退回
    正文。**分数只从正文解析**：推理段里全是数字（「资料2」「2025 年」），拿它解析
    会把资料编号当成分数。
    """
    content = reasoning = ""
    try:
        content, reasoning = llm.chat_with_reasoning([{"role": "user", "content": prompt}])
        content = content.strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("judge 调用失败，取 0.5: %s", exc)
        return 0.5, f"<调用失败: {exc}>"
    reason = (reasoning or content).strip()
    # 别用 `float(content)` 整串强转：弱模型可能输出 "0.7"、"评分 0.7"、"0.7 分" 甚至
    # 一段带数字的解释，整串转会抛异常、或把 "0" 当成合法分数静默通过。这里只取
    # 第一个数字；数字都取不到才兜底 0.5 并留痕。
    m = _FLOAT_RE.search(content)
    if m is None:
        log.warning("judge 没输出数字，取 0.5（正文：%r）", content[:200])
        return 0.5, reason
    return max(0.0, min(1.0, float(m.group(0)))), reason


def _clip(text: str, limit: int = 220) -> str:
    """把 judge 理由压成一行便于打印：换行折叠成空格，超长截断加省略号。"""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _warn_if_constant(label: str, scores: list[float]) -> None:
    """judge 对所有样例输出同一个分数 = 大概率没真的打分，结果不可信，要显式告警。"""
    if scores and len({round(s, 6) for s in scores}) == 1:
        log.warning("%s 恒为 %.3f，judge 可能没真的打分，结果不可信", label, scores[0])


def judge_faithfulness(llm: LLMClient, question: str, answer: str, contexts: list[dict]) -> tuple[float, str]:
    """评「答案是否忠于检索到的上下文」，返回 (分数, 理由)。

    注意它**不评答案对不对**：没有 ground truth 参与，所以答案照抄一段错误的上下文
    也能拿满分。这正是本仓要的两件事之一——faithfulness 管「不许编」，正确性靠
    QA_SAMPLES 的 reference 人工核对。这也是 reference 不进 prompt 的原因。
    """
    ctx = "\n".join(f"- {c['text']}" for c in contexts)
    prompt = (
        "请判断以下回答是否完全基于给定上下文、没有编造事实。只输出 0~1 的小数"
        "（1=完全忠实，0=完全不忠实），不要输出任何解释或额外文字。\n"
        f"问题：{question}\n回答：{answer}\n上下文：\n{ctx}\n"
    )
    return _ask_score(llm, prompt)


def judge_context_relevancy(llm: LLMClient, question: str, contexts: list[dict]) -> tuple[float, str]:
    """评「检索上下文与问题是否相关」，返回 (分数, 理由)。

    只看问题与上下文、**不看答案**，所以它衡量的是检索/重排这一段的质量：答案错但
    检索对了，faithfulness 会掉、这一项不会。两项一起看才能定位问题出在检索还是生成。
    """
    ctx = "\n".join(f"- {c['text']}" for c in contexts)
    prompt = (
        "请判断以下检索上下文与问题的相关程度。只输出 0~1 的小数"
        "（1=高度相关，0=无关），不要输出任何解释或额外文字。\n"
        f"问题：{question}\n上下文：\n{ctx}\n"
    )
    return _ask_score(llm, prompt)


def main() -> None:
    """按 QA_SAMPLES 逐条跑「检索 → 生成 → judge」并打印分数，无返回值、无退出码语义。

    失败不影响流程：judge 调不通时 _ask_score 兜 0.5 并留痕，评测会跑完全部样例，
    所以「某条 0.5」要先看日志里有没有 judge 调用失败，别直接当成模型表现。
    """
    setup_logging(settings)
    ap = argparse.ArgumentParser(description="RAGAS 风格评测（简化版）")
    ap.add_argument("--role", default="psychologist", choices=sorted(QA_SAMPLES),
                    help="评测哪个角色（决定用哪组题；默认与 schemas.ChatRequest 的默认角色一致）")
    # judge 与答案模型解耦：答案用 .env 的 LLM_MODEL，judge 单独指定。
    # 弱模型当 judge 会恒输出 0（faithfulness 假性挂零），默认换成更强的 8b。
    ap.add_argument("--judge-model", default="qwen3:8b",
                    help="judge 用的模型（默认 qwen3:8b）")
    args = ap.parse_args()

    pipeline = RAGPipeline.build(settings)
    # 单独建 judge 客户端：换模型 + 温度归零（judge 要确定性，不要采样随机）
    # replace 是 dataclass 克隆：只覆盖这两个字段，base_url/api_key 等仍沿用 .env，
    # 且**不动全局 settings 单例**——改了它，同一个进程里的被测 pipeline 也会跟着变
    judge = LLMClient(replace(settings, llm_model=args.judge_model, llm_temperature=0.0))

    results = []
    # 按角色取题：--role 已限定取值，这里必定命中（choices 就是 QA_SAMPLES 的键）
    for idx, qa in enumerate(QA_SAMPLES[args.role]):
        q = qa["question"]
        # 显式调 retrieve 单独给 judge 用：answer 虽然也会检索，但它返回的 sources 是
        # 经 to_sources 处理过的（分数被 round 到 4 位、丢了 summary 等字段），judge 该看
        # 检索的原样输出。代价是每条样例检索两次——评测场景下这个开销可以接受。
        chunks = pipeline.retrieve(q, args.role)
        # 每条样例独立 session：共用一个 session 会让后面的问题带上前面的问答历史，
        # 污染多轮上下文，评测分数不再可比
        out = pipeline.answer(q, args.role, session_id=f"eval-{idx}")
        faith, faith_raw = judge_faithfulness(judge, q, out["answer"], chunks)
        relev, relev_raw = judge_context_relevancy(judge, q, chunks)
        results.append({
            "question": q,
            "faithfulness": faith, "faithfulness_reason": faith_raw,
            "context_relevancy": relev, "context_relevancy_reason": relev_raw,
        })
        print(f"Q: {q}")
        print(f"  faithfulness={faith:.2f}  context_relevancy={relev:.2f}")
        print(f"  answer: {out['answer'][:80]}...")
        # judge 的理由一并打出来：分数低时，只有看理由才能分清是「judge 误判」还是
        # 「模型真的在编」。不给理由的话，faithfulness=0 这种结论没法复核。
        print(f"  judge(faithfulness): {_clip(faith_raw)}")
        print(f"  judge(relevancy):    {_clip(relev_raw)}\n")

    # 打印平均分之前先查恒值：恒值时这个平均值是个无意义的数，告警必须出现在它前面，
    # 否则读者先看到「平均 0.800」就不会再往下看告警了
    if results:
        _warn_if_constant("faithfulness", [r["faithfulness"] for r in results])
        _warn_if_constant("context_relevancy", [r["context_relevancy"] for r in results])
        avg_f = sum(r["faithfulness"] for r in results) / len(results)
        avg_r = sum(r["context_relevancy"] for r in results) / len(results)
        print(f"平均 faithfulness={avg_f:.3f}，context_relevancy={avg_r:.3f}（共 {len(results)} 条）")


if __name__ == "__main__":
    main()
