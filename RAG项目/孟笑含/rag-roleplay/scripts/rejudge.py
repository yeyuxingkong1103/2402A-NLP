# -*- coding: utf-8 -*-
"""用另一个 judge 大模型重判已生成的评测样本（不重新生成，只重判）。

输出 eval_report/final_anchor.md：新 judge 的分数 + 与原 judge 的对比。

用法：
    python scripts/rejudge.py 报告.json [输出路径] [--model 模型名]
judge 配置优先级：--model 参数 > .env 的 JUDGE_* > 回落到 LLM_*（复用生成模型同 key）
"""
import json
# 解析：JSON（读写报告）
import logging
# 解析：日志
import sys
# 解析：命令行参数
from pathlib import Path
# 解析：路径

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# 解析：日志配置
logger = logging.getLogger("rag-rejudge")
# 解析：重判 logger

ROOT = Path(__file__).resolve().parent.parent
# 解析：项目根
sys.path.insert(0, str(ROOT))
# 解析：加导入路径

from app.config import settings  # noqa: E402
# 解析：配置
from app.eval.report import build_comparison  # noqa: E402
# 解析：对比报告
from app.eval.dataset import load_samples_from_report  # noqa: E402
# 解析：从报告取样本
from app.eval.ragas_metrics import LocalEmbeddings, build_judge_llm, run_ragas  # noqa: E402
# 解析：RAGAS 封装
from app.eval.report import format_report  # noqa: E402
# 解析：报告格式化
from app.rag.models import BGEM3Embedder  # noqa: E402
# 解析：本地向量化模型


# 用另一个 judge 大模型重判已生成样本（不重新生成）
def main() -> None:
    # 解析：重判主流程
    argv = sys.argv[1:]
    # 解析：命令行参数
    model_override = None
    # 解析：模型覆盖变量
    if "--model" in argv:  # --model 模型名
        # 解析：空格分隔形式
        model_override = argv[argv.index("--model") + 1]
        # 解析：取下个参数为模型名
        argv = [a for a in argv if a not in ("--model", model_override)]
        # 解析：从参数列表移除
    for a in list(argv):  # --model=模型名
        # 解析：等号形式
        if a.startswith("--model="):
            # 解析：匹配
            model_override = a.split("=", 1)[1]
            # 解析：取等号后
            argv.remove(a)
            # 解析：移除
    args = argv
    # 解析：剩余位置参数
    report_path = Path(args[0]) if args else None
    # 解析：报告路径（第一个位置参数）
    if report_path is None:
        # 解析：没给报告
        raise SystemExit("用法：python scripts/rejudge.py 报告.json [输出路径] [--model 模型名]")
        # 解析：退出并提示用法
    out_path = Path(args[1]) if len(args) > 1 else ROOT / "eval_report" / "final_anchor.md"
    # 解析：输出路径（缺省锚点报告）

    # 配置优先级：--model 参数 > JUDGE_* > 回落 LLM_*
    judge_base_url = settings.judge_base_url or settings.llm_base_url
    # 解析：judge 地址（回落生成模型地址）
    judge_api_key = settings.judge_api_key or settings.llm_api_key
    # 解析：judge 密钥（回落生成密钥）
    judge_model = model_override or settings.judge_model or settings.llm_model
    # 解析：judge 模型（参数 > 配置 > 回落）

    samples = load_samples_from_report(report_path)
    # 解析：加载已生成样本（不重新生成，省 API 费用）
    logger.info("加载已生成样本 %d 条（不重新生成）", len(samples))
    # 解析：日志

    judge = build_judge_llm(judge_base_url, judge_api_key, judge_model)
    # 解析：构造新 judge
    logger.info("judge 模型：%s", judge_model)
    # 解析：日志

    embedder = BGEM3Embedder(settings.embedding_model_path)
    # 解析：加载本地 BGE-m3
    scores, rows = run_ragas(samples, judge, LocalEmbeddings(embedder))
    # 解析：用新 judge 重新判分

    # 新 judge 主报告
    md = format_report(
        # 解析：生成主报告
        f"最终锚点评测（judge: {judge_model}）",
        # 解析：标题带 judge 名
        "林医生",
        # 解析：角色
        scores,
        # 解析：分数
        rows,
        # 解析：明细
        model=judge_model,
        # 解析：模型
    )

    # 与原 judge（报告 JSON 里的 scores）对比
    original = json.loads(report_path.read_text(encoding="utf-8"))
    # 解析：读原报告 JSON
    comparison = build_comparison(
        # 解析：生成对比
        base_name=f"judge: 原报告({report_path.stem})",
        # 解析：基线名
        cur_name=f"judge: {judge_model}",
        # 解析：本轮名
        base_scores=original.get("scores", {}),
        # 解析：原分数
        base_rows=original.get("rows", []),
        # 解析：原明细
        cur_scores=scores,
        # 解析：新分数
        cur_rows=rows,
        # 解析：新明细
    )
    final_md = md.rstrip() + "\n\n---\n\n## 与原 judge 的对比（注意：同一批回答，不同 judge 打分）\n\n" + comparison
    # 解析：拼接主报告与对比段

    out_path.write_text(final_md, encoding="utf-8")
    # 解析：保存 markdown
    json_path = out_path.with_suffix(".json")
    # 解析：同名 JSON
    json_path.write_text(
        # 解析：保存 JSON
        json.dumps({"judge_model": judge_model, "scores": scores, "rows": rows},
                   ensure_ascii=False, indent=2),
        # 解析：judge、分数、明细
        encoding="utf-8",
        # 解析：UTF-8
    )
    logger.info("已保存：%s / %s", out_path, json_path)
    # 解析：日志
    print(final_md)
    # 解析：终端输出


if __name__ == "__main__":
    # 解析：入口
    main()
    # 解析：执行
