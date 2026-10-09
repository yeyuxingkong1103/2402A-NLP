import json, os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.format_stats import format_distribution
from core.pdf_classifier import classify_pdf
from core.length_dist import length_distribution
from core.duplicate_detector import md5_duplicates, simhash_similar
from core.sensitive_scanner import scan_sensitive
from core.classifier import classify_document
from utils.file_utils import iter_files

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _load_ckpt(p):
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"done": [], "pdf_results": {}}

def _save_ckpt(p, done, pdf_results):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"done": list(done), "pdf_results": pdf_results}, f, ensure_ascii=False)
    os.replace(tmp, p)

def assess(folder_path, config_path):
    cfg = load_config(config_path)
    ckpt_path = cfg["paths"]["checkpoint_file"]
    ckpt = _load_ckpt(ckpt_path)
    done = set(ckpt["done"])
    pdf_results = ckpt["pdf_results"]

    files = [str(p) for p in iter_files(folder_path)]
    total = len(files)
    pdf_files = [f for f in files if Path(f).suffix.lower() == ".pdf"]

    todo = [f for f in pdf_files if f not in done]
    print("[info] total=%d, pdf=%d, todo=%d" % (total, len(pdf_files), len(todo)), flush=True)
    with ThreadPoolExecutor(max_workers=cfg["performance"]["max_workers"]) as ex:
        futs = {ex.submit(classify_pdf, Path(f), cfg): f for f in todo}
        for i, fut in enumerate(as_completed(futs)):
            f = futs[fut]
            try:
                pdf_results[f] = fut.result()
            except Exception as e:
                pdf_results[f] = {"type": "error", "error": str(e)}
            done.add(f)
            if (i+1) % cfg["performance"]["progress_interval"] == 0:
                _save_ckpt(ckpt_path, done, pdf_results)
                print("[checkpoint] %d/%d" % (i+1, len(todo)), flush=True)
    _save_ckpt(ckpt_path, done, pdf_results)

    fmt = format_distribution(folder_path)
    length = length_distribution(files, cfg)
    md5_rep = md5_duplicates(files) if cfg["duplicate"]["enable_md5"] else {}
    sim_rep = simhash_similar(files, cfg) if cfg["duplicate"]["enable_simhash"] else {}
    sens = scan_sensitive(files, cfg)

    labels = {}
    for f in files:
        labels[f] = classify_document(Path(f), pdf_results.get(f, {}), cfg)

    to_confirm = {
        "pdf_mixed": [f for f, r in pdf_results.items() if r.get("need_confirm")],
        "version_conflicts": sim_rep.get("pairs", []),
        "sensitive_items": sens.get("items", []),
    }

    report = {
        "summary": {
            "folder": folder_path, "total_files": total, "pdf_files": len(pdf_files),
            "scan_pdf": sum(1 for r in pdf_results.values() if r.get("type") == "scan"),
            "text_pdf": sum(1 for r in pdf_results.values() if r.get("type") == "text"),
            "mixed_pdf": sum(1 for r in pdf_results.values() if r.get("type") == "mixed"),
        },
        "format_distribution": fmt,
        "pdf_classification": pdf_results,
        "length_distribution": length,
        "duplicate_report": {"md5": md5_rep, "simhash": sim_rep},
        "sensitive_report": sens,
        "classification_labels": labels,
        "to_confirm": to_confirm,
    }
    return report

if __name__ == "__main__":
    folder = sys.argv[1]
    cfg_path = sys.argv[2] if len(sys.argv) > 2 else "assessment_config.yaml"
    rep = assess(folder, cfg_path)
    out = load_config(cfg_path)["paths"]["output_dir"]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print("[done] report -> %s/report.json" % out)
