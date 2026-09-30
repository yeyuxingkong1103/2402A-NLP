# -*- coding: utf-8 -*-
"""对比两轮 RAGAS 评测结果，输出 eval_report/comparison.md。

用法：
    python scripts/compare_eval.py 基线.json 第二轮.json [输出路径]
"""
import json
# 解析：JSON 读写
import sys
# 解析：命令行参数
from pathlib import Path
# 解析：路径

ROOT = Path(__file__).resolve().parent.parent
# 解析：项目根
sys.path.insert(0, str(ROOT))
# 解析：加导入路径

from app.eval.report import build_comparison  # noqa: E402
# 解析：对比报告函数


# 对比两轮 RAGAS 评测 JSON，输出 markdown 对比报告
def main() -> None:
    # 解析：对比主流程
    if len(sys.argv) < 3:
        # 解析：参数不足
        raise SystemExit("用法：python scripts/compare_eval.py 基线.json 本轮.json [输出路径]")
        # 解析：提示用法
    base_path, cur_path = Path(sys.argv[1]), Path(sys.argv[2])
    # 解析：两轮报告路径
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "eval_report" / "comparison.md"
    # 解析：输出路径（缺省 comparison.md）

    base = json.loads(base_path.read_text(encoding="utf-8"))
    # 解析：读基线报告
    cur = json.loads(cur_path.read_text(encoding="utf-8"))
    # 解析：读本轮报告
    md = build_comparison(
        # 解析：生成对比
        base_name=base_path.stem,
        # 解析：基线名
        cur_name=cur_path.stem,
        # 解析：本轮名
        base_scores=base["scores"],
        # 解析：基线分数
        base_rows=base["rows"],
        # 解析：基线明细
        cur_scores=cur["scores"],
        # 解析：本轮分数
        cur_rows=cur["rows"],
        # 解析：本轮明细
    )
    out_path.write_text(md, encoding="utf-8")
    # 解析：保存报告
    print(md)
    # 解析：打印
    print(f"[已保存] {out_path}")
    # 解析：提示


if __name__ == "__main__":
    # 解析：入口
    main()
    # 解析：执行
