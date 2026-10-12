# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估 —— 量 RAGAS 的噪声底（比较"改前改后"之前的强制动作）
# 依据：全局纪律「不替未跑过的数字背书」+ 历史经验「比较两轮指标前必须先量噪声底」。
# 做法：把某轮已落盘的 raw_*.json（含 question/answer/contexts/ground_truth）**原样**重打分一次。
#       输入逐字未变，若分值仍有波动，该波动 100% 来自判官的非确定性 —— 这就是噪声底。
# 用法：python scripts/measure_noise.py [raw_*.json]
import sys, json, pathlib, statistics                                 # 读文件、统计
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # 项目根
from app import config as C                                           # 配置
from app.evaluate import ragas_metrics                                # 复用同一条评估实现，不另写口径

def main():                                                           # 入口
    f = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((C.RESULTS).glob("raw_*.json"))[0]  # 待重打分的原始输入
    rows = json.loads(f.read_text(encoding="utf-8"))                  # 该轮的逐题原始输入
    print(f"重打分：{f.name}（{len(rows)} 题，输入逐字未改）")            # 显式说明输入没动
    again = ragas_metrics(rows)                                       # 第二遍打分
    (C.RESULTS / "noise_reread.json").write_text(json.dumps(again, ensure_ascii=False, indent=1), encoding="utf-8")  # 落盘
    print("第二遍:", {k: v for k, v in again.items() if not k.startswith("per_sample_")})  # 打印
    return again                                                      # 返回

if __name__ == "__main__":                                            # 直接运行
    main()
