# -*- coding: utf-8 -*-
# eval_vlm.py 工单16 VLM评估脚本：通用BLEU/ROUGE + 工业专项指标
import json
from evaluate import load

bleu = load("bleu")
rouge = load("rouge")

def calc_industrial_metrics(pred_list, ref_list, term_vocab_path="./dataset/industrial_terms.txt"):
    """
    工单重点：工业专项指标
    1.专业术语准确率：标准答案中工业术语是否出现在模型预测
    2.图纸推理正确率：简单实现，匹配图纸相关关键部件、位置关键词
    """
    # 加载工业术语词表
    if os.path.exists(term_vocab_path):
        with open(term_vocab_path,"r",encoding="utf‑8") as f:
            term_vocab = [l.strip() for l in f.readlines() if l.strip()]
    else:
        term_vocab = ["部件","图纸","位置","编号","装置","链条","外壳","盘","阀"]

    total_terms = 0
    hit_terms = 0
    draw_reason_correct = 0
    draw_total = 0

    for pred,ref in zip(pred_list,ref_list):
        # 术语统计
        for t in term_vocab:
            if t in ref:
                total_terms +=1
                if t in pred:
                    hit_terms +=1
        # 图纸推理简易判断：ref包含图纸关键词
        if "图" in ref or "部件" in ref:
            draw_total +=1
            # 简易判断：关键信息是否存在
            if any(w in pred for w in ref.split()):
                draw_reason_correct +=1

    term_acc = hit_terms / total_terms if total_terms>0 else 0.0
    draw_acc = draw_reason_correct / draw_total if draw_total>0 else 0.0
    return {
        "term_accuracy":round(term_acc,4),
        "draw_reasoning_accuracy":round(draw_acc,4)
    }

def evaluate_one_group(preds,refs):
    """一组预测+标准答案，计算全部指标"""
    res_bleu = bleu.compute(predictions=preds, references=[[r] for r in refs])
    res_rouge = rouge.compute(predictions=preds, references=[[r] for r in refs])
    res_industrial = calc_industrial_metrics(preds,refs)
    out = {
        "bleu‑4":round(res_bleu["bleu"],4),
        "rouge‑l":round(res_rouge["rougeL"],4),
        **res_industrial
    }
    return out

if __name__ == "__main__":
    # 模拟数据，实际替换成基线、微调模型输出结果
    baseline_pred = ["部件13在部件12之内","顶部","部件13"]
    ft_pred = ["位于编号12的部件之内","顶部位置","下一步经过部件13"]
    ground_truth = ["位于编号12的部件之内","顶部","部件13"]

    print("====基线原始VLM评估====")
    base_result = evaluate_one_group(baseline_pred,ground_truth)
    print(json.dumps(base_result,indent=2,ensure_ascii=False))

    print("\n====LoRA微调后VLM评估====")
    ft_result = evaluate_one_group(ft_pred,ground_truth)
    print(json.dumps(ft_result,indent=2,ensure_ascii=False))

    final_report = {
        "baseline":base_result,
        "lora_finetuned":ft_result
    }
    with open("./vlm_eval_report.json","w",encoding="utf‑8") as f:
        json.dump(final_report,f,ensure_ascii=False,indent=2)
    print("\n✅评估报告输出 vlm_eval_report.json")
