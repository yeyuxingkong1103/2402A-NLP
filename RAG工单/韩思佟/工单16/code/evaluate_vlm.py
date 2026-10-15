# -*- coding: utf-8 -*-
"""
工单16 —— 专用视觉语言模型评估脚本

评估三类指标（对应工单「实施专业评估」要求）：
  a) 通用文本指标：BLEU-1/2/4、ROUGE-L（本文件内置实现，不依赖 nltk/rouge 包）
  b) 专业术语准确性：答案中的工业术语是否命中参考答案中的术语（TERM_HINTS + 自动抽取）
  c) 图纸推理正确性：对含"图示/部件编号/位置关系"的题目，检查关键要素是否出现
另外可用 --judge 打开 LLM-as-Judge，用大模型给 0~1 的正确性分。

后端二选一：
  --backend local   : 用 transformers 直接加载（基座或合并后的模型）
  --backend api     : 调 OpenAI 兼容的多模态接口（推荐，省显存）

用法：
  python evaluate_vlm.py --backend api  --model Qwen/Qwen2-VL-2B-Instruct --tag base
  python evaluate_vlm.py --backend api  --model saves/xxx/merged        --tag ft
  python evaluate_vlm.py --backend local --model Qwen/Qwen2-VL-2B-Instruct --tag base --limit 30
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

TERM_HINTS = ["淬火", "公差配合", "退火", "回火", "渗碳", "轴承", "齿轮", "螺栓",
              "阀", "泵", "轴", "壳体", "密封", "润滑", "力矩", "浇注", "模具",
              "除尘器", "筛板", "带孔盘", "圆锥", "法兰", "焊接", "气流", "配气"]
VISUAL_KEYS = ["图", "部件", "编号", "左侧", "右侧", "上方", "下方", "顶部", "底部",
               "圆锥", "方向", "间距"]


# ---------------------------------------------------------------- 文本指标
def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(pred: str, ref: str, max_n: int = 4) -> dict:
    p_tok = list(pred.strip())
    r_tok = list(ref.strip())
    out = {}
    for n in range(1, max_n + 1):
        p, r = ngrams(p_tok, n), ngrams(r_tok, n)
        if not p:
            out[f"BLEU-{n}"] = 0.0
            continue
        overlap = sum((p & r).values())
        out[f"BLEU-{n}"] = overlap / max(sum(p.values()), 1)
    # 简单几何平均（不做长度惩罚，便于对比相对变化）
    vals = [out[f"BLEU-{n}"] for n in range(1, max_n + 1)]
    out["BLEU-avg"] = (math.prod(vals) ** (1 / max_n)) if all(v > 0 for v in vals) else 0.0
    return out


def lcs(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(pred: str, ref: str) -> float:
    a, b = list(pred.strip()), list(ref.strip())
    if not a or not b:
        return 0.0
    l = lcs(a, b)
    prec, rec = l / len(a), l / len(b)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------- 专项指标
def term_accuracy(pred: str, ref: str) -> tuple:
    """专业术语准确性：参考答案里出现的工业术语，有多少在预测里也出现"""
    ref_terms = {t for t in TERM_HINTS if t in ref}
    if not ref_terms:
        return 1.0, []
    hit = [t for t in ref_terms if t in pred]
    return len(hit) / len(ref_terms), sorted(ref_terms - set(hit))


def is_visual_question(q: str, ref: str) -> bool:
    return any(k in q + ref for k in ["图", "部件", "编号"])


def visual_reasoning_accuracy(pred: str, ref: str) -> tuple:
    """图纸推理正确性：答案里的位置/方向/编号要素是否与参考答案一致"""
    ref_keys = {k for k in VISUAL_KEYS if k in ref}
    ref_nums = set(re.findall(r"\d+", ref))
    if not ref_keys and not ref_nums:
        return None, []
    hit_keys = [k for k in ref_keys if k in pred]
    hit_nums = [n for n in ref_nums if n in pred]
    total = len(ref_keys) + len(ref_nums)
    return (len(hit_keys) + len(hit_nums)) / max(total, 1), {
        "缺失要素": sorted(set(ref_keys) - set(hit_keys)),
        "缺失编号": sorted(ref_nums - set(hit_nums)),
    }


# ---------------------------------------------------------------- 后端
class ApiBackend:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        from src import config
        self.model = model
        self.base_url = (base_url or os.getenv("VLM_BASE_URL") or config.LLM_BASE_URL).rstrip("/")
        self.api_key = api_key or os.getenv("VLM_API_KEY") or config.LLM_API_KEY

    def ask(self, image_path: str, question: str) -> str:
        import urllib.request
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
            ]}],
            "max_tokens": 512, "temperature": 0.0,
        }
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.api_key}, method="POST")
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode())["choices"][0]["message"]["content"].strip()


class LocalBackend:
    def __init__(self, model: str):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        self.torch, self.pvi = torch, process_vision_info
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model, torch_dtype=torch.bfloat16, device_map="auto")
        self.proc = AutoProcessor.from_pretrained(model)

    def ask(self, image_path: str, question: str) -> str:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{Path(image_path).as_posix()}"},
            {"type": "text", "text": question}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_in, _ = self.pvi(msgs)
        inputs = self.proc(text=[text], images=img_in, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        return self.proc.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                      skip_special_tokens=True)[0].strip()


class JudgeBackend:
    """用文本大模型做 0~1 正确性打分（复用 DeepSeek）"""

    def __init__(self):
        from src.llm import get_llm
        self.llm = get_llm()

    def score(self, q: str, pred: str, ref: str) -> float:
        prompt = (f"问题：{q}\n参考答案：{ref}\n模型答案：{pred}\n"
                  '请判断模型答案在事实上是否与参考答案一致，只输出 JSON：{"score":0~1}')
        try:
            d = self.llm.ask_json(prompt, max_tokens=120)
            return float(d.get("score", 0) or 0)
        except Exception:                                        # noqa: BLE001
            return 0.0


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="api", choices=["api", "local"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--img-dir", default=None)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent.parent
    test_path = Path(args.test) if args.test else here / "产出物" / "vlm_sft_sharegpt.jsonl"
    if not test_path.exists():
        print(f"测试集不存在：{test_path}，请先运行 build_vlm_dataset.py")
        sys.exit(1)
    items = [json.loads(l) for l in test_path.read_text(encoding="utf-8").splitlines()
             if l.strip()][: args.limit]

    backend = ApiBackend(args.model) if args.backend == "api" else LocalBackend(args.model)
    judge = JudgeBackend() if args.judge else None

    rows = []
    for i, it in enumerate(items, 1):
        if "messages" in it:
            q = it["messages"][0]["content"].replace("<image>", "").strip()
            ref = it["messages"][1]["content"]
            img = (it.get("images") or [""])[0]
        else:
            q, ref, img = it["instruction"], it["output"], it.get("image", "")
        if not img or not Path(img).exists():
            continue
        try:
            pred = backend.ask(img, q)
        except Exception as e:                                   # noqa: BLE001
            pred = f"[ERROR] {e}"
        b = bleu(pred, ref)
        ta, missing_terms = term_accuracy(pred, ref)
        vr, vr_detail = visual_reasoning_accuracy(pred, ref) if is_visual_question(q, ref) else (None, None)
        row = {"i": i, "question": q, "ref": ref, "pred": pred,
               **b, "ROUGE-L": round(rouge_l(pred, ref), 4),
               "术语准确性": round(ta, 3), "缺失术语": missing_terms,
               "图纸推理": None if vr is None else round(vr, 3),
               "图纸缺失": vr_detail}
        if judge:
            row["judge_score"] = round(judge.score(q, pred, ref), 3)
        rows.append(row)
        if i % 5 == 0:
            print(f"  {i}/{len(items)}  BLEU-avg={row['BLEU-avg']:.3f} "
                  f"ROUGE-L={row['ROUGE-L']:.3f} 术语={row['术语准确性']:.2f}")

    def avg(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary = {f"平均{k}": avg(k) for k in
               ["BLEU-1", "BLEU-2", "BLEU-4", "BLEU-avg", "ROUGE-L",
                "术语准确性", "图纸推理"]}
    if args.judge:
        summary["平均judge_score"] = avg("judge_score")
    summary["样本数"] = len(rows)

    outdir = here / "产出物"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"评估_{args.tag}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
