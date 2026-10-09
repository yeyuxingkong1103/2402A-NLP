import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import assess, load_config

if __name__ == "__main__":
    folder = sys.argv[1]
    cfg_path = sys.argv[2] if len(sys.argv) > 2 else "assessment_config.yaml"
    rep = assess(folder, cfg_path)
    out = load_config(cfg_path)["paths"]["output_dir"]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print("[done] report -> %s/report.json" % out)
