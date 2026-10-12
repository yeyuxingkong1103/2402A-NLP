# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估 —— 评估入口脚本（核心逻辑在 app/evaluate.py）
# 用法：python scripts/run_eval.py [题目数limit] [--no-ragas]
import sys, pathlib                                              # 参数与环境准备
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # 把项目根加进模块搜索路径
from app.evaluate import run                                     # 导入评估主流程
if __name__ == "__main__":                                       # 直接运行本文件即执行评估
    a = [x for x in sys.argv[1:] if not x.startswith("--")]      # 位置参数：限定题数
    limit = int(a[0]) if a else None                             # 不传则跑全部 10 题
    res = run(limit=limit, skip_ragas="--no-ragas" in sys.argv)   # --no-ragas 时只跑检索与生成，先看检索效果
    print(f"共 {res['n']} 题，失败 {res['errors']} 题，p50 {res['p50_ms']}ms，max {res['max_ms']}ms")  # 汇总
    print("RAGAS:", {k: v for k, v in res["ragas"].items() if not k.startswith("per_sample_")})  # 打印指标（含有效样本数）
