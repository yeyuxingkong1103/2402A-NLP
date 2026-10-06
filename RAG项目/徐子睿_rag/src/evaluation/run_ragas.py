"""evaluation/run_ragas.py —— 新架构的评测报告入口（当前为占位实现）。

在链路中的位置：
    独立脚本。读 evaluation/dataset.json，产出 evaluation/reports/latest.json。

用法：
    python evaluation/run_ragas.py

当前的诚实状态（这一点很重要）：
    本脚本**还没有真正接入 RAGAS**。它写出的是一个 status="placeholder" 的
    占位报告，明确说明"安装 ragas 并接入真实 answers 后可计算
    Faithfulness/AnswerRelevancy/ContextPrecision/ContextRecall"。
    也就是说：它不会给出一组看着漂亮、实则没有依据的分数。

    这种"宁可显式声明是占位，也不输出假指标"的做法，
    与本项目一贯的原则一致，也是评测工作最该守住的底线 ——
    一个假分数会误导后续所有的优化决策。
"""
from __future__ import annotations

import json
from pathlib import Path


def role_consistency(answer: str, role_id: str) -> float:
    """用关键词启发式判断回答是否保持了角色特征（角色一致性）。

    参数：
        answer: 模型生成的回答
        role_id: 该回答所属的角色
    返回：
        1.0 = 命中角色特征词；0.5 = 未命中；0.7 = 未知角色的默认值。

    为什么要有这个指标：
        RAGAS 的 faithfulness / answer_relevancy 衡量的是"答案忠实于上下文"
        和"答案切题"，但都不管"这句话像不像这个角色说的"。
        而角色扮演系统的核心体验恰恰在后者 —— 律师该讲依据和风险，
        心理咨询师该讲感受和支持。

    当前实现是关键词匹配，明确标注为启发式：
        它能抓住"跑题成完全不同的角色"这类明显问题，
        但识别不了微妙的人设漂移（措辞变但不换领域）。
        报告里写明"生产可替换为 LLM-as-judge"，指出了升级路径。

    未命中给 0.5 而不是 0：
        关键词没出现不等于角色不对（可能只是换了种说法），
        给 0 会过度惩罚；给 1 又会掩盖问题。0.5 表达"不确定"。

    未知角色返回 0.7：
        这是一个中间偏正的默认值 —— 没有该角色的评判标准时，
        不做苛刻判断，也不给满分。
    """
    if role_id == "lawyer":
        return 1.0 if any(word in answer for word in ["依据", "法律", "风险", "律师"]) else 0.5
    if role_id == "psychologist":
        return 1.0 if any(word in answer for word in ["感受", "支持", "求助", "呼吸"]) else 0.5
    return 0.7


def main() -> int:
    """生成评测报告（当前为占位实现）。

    返回：
        0（成功）。

    两个流程细节：
        1. dataset.json 不存在时先调用 build_dataset 生成 ——
           让本脚本可以独立运行，不必记住"要先跑数据集构建"
        2. 报告写到 reports/ 子目录，写前先 mkdir(parents=True, exist_ok=True) ——
           首次运行时该目录不存在，不建会直接报错

    import build_dataset 用的是模块名而不是相对导入：
        本脚本预期以 `python evaluation/run_ragas.py` 方式运行，
        此时 evaluation/ 目录在 sys.path 上，模块名导入能正确解析。

    报告必须在 note 字段里说明这是占位：
        一个被保存下来的 JSON 文件很容易脱离上下文被引用，
        文件自身带上"我是占位"的声明，才不会被后来的人误当成真实评测结果。
    """
    dataset_path = Path(__file__).with_name("dataset.json")
    if not dataset_path.exists():
        import build_dataset

        build_dataset.main()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    report = {"status": "placeholder", "note": "安装 ragas 并接入真实 answers 后可计算 Faithfulness/AnswerRelevancy/ContextPrecision/ContextRecall", "samples": len(dataset), "role_consistency_formula": "关键词启发式，生产可替换为 LLM-as-judge"}
    out = Path(__file__).parent / "reports" / "latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
