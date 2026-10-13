# 工单编号：人工智能 NLP-RAG-LightRAG 优化
import json
import re
import jieba
from config_v12 import (
    COMPARE_RESULT_FILE, EVAL_RESULT_FILE, RAG_RESULT_FILE
)

GROUND_TRUTH = {
    "5": "销售部4个部门 渠道销售部 电话及网络销售部 大客户销售部 国际贸易部 6个销售处 北京 广州 成都 深圳 武汉 珠海",
    "6": "汽车行业14.0% IC卡行业-2.0%",
    "1": "1,670万股 25.04%",
    "2": "仓储及物流中心 研发中心 电子商务平台 扩充产品种类和数量",
    "3": "赵马克 42.35% 控股股东",
    "4": "融冰投资 武汉博润 上海博润 听音投资 联众聚源 力源贸易 普芯达",
    "260": "6,464.51 14,414.16 18,780.67 4,627.14",
    "95": "全军第一个视频指挥系统技术标准",
    "33": "82.10 97.31 94.84 94.34",
    "34": "电子元器件制造业 金属制品业",
    "957": "军队视频指挥控制",
    "793": "军队 企事业单位",
    "795": "情报 指挥 控制 通信网络一体化工程",
    "543": "5,520.00",
    "531": "程家明",
    "207": "15,000.00",
}


def extract_keywords(text):
    nums = re.findall(r"\d[\d,\.]*", text)
    words = [w for w in jieba.cut(text) if len(w) >= 2]
    return set(nums) | set(words)


def compute_p_r(context, gt):
    gt_keys = extract_keywords(gt)
    if not gt_keys:
        return 0.0, 0.0
    hit = sum(1 for k in gt_keys if k in context)
    score = hit / len(gt_keys)
    return score, score


if __name__ == "__main__":
    with open(COMPARE_RESULT_FILE, "r", encoding="utf-8") as f:
        light_results = json.load(f)

    try:
        with open(RAG_RESULT_FILE, "r", encoding="utf-8") as f:
            rag_results = {str(r["id"]): r for r in json.load(f)}
        print(f"加载 {len(rag_results)} 条 RAG 结果")
    except FileNotFoundError:
        rag_results = {}
        print("无 RAG 结果文件")

    light_metrics = []
    rag_metrics = []

    for lr in light_results:
        qid = str(lr["id"])
        gt = GROUND_TRUTH.get(qid, "")

        l_ctx = lr["lightrag_context"]
        lp, lr_recall = compute_p_r(l_ctx, gt)
        light_metrics.append({"id": qid, "P": lp, "R": lr_recall})
        print(f"[{qid}] LightRAG P={lp:.2f} R={lr_recall:.2f}")

        if qid in rag_results:
            r = rag_results[qid]
            r_ctx = r.get("rag_answer_v3", r.get("rag_answer", r.get("answer", "")))
            rp, rr = compute_p_r(r_ctx, gt)
            rag_metrics.append({"id": qid, "P": rp, "R": rr})
            print(f"       RAG      P={rp:.2f} R={rr:.2f}")
        else:
            print(f"       RAG      无数据")


    def avg(metrics, key):
        if not metrics:
            return 0.0
        return sum(m[key] for m in metrics) / len(metrics)

    result = {
        "lightrag": {
            "avg_precision": round(avg(light_metrics, "P"), 3),
            "avg_recall": round(avg(light_metrics, "R"), 3),
            "details": light_metrics
        },
        "rag": {
            "avg_precision": round(avg(rag_metrics, "P"), 3),
            "avg_recall": round(avg(rag_metrics, "R"), 3),
            "details": rag_metrics
        }
    }

    print("\n===== 对比结果 =====")
    print(f"LightRAG: P={result['lightrag']['avg_precision']}, R={result['lightrag']['avg_recall']}")
    print(f"RAG:      P={result['rag']['avg_precision']}, R={result['rag']['avg_recall']}")

    with open(EVAL_RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {EVAL_RESULT_FILE}")