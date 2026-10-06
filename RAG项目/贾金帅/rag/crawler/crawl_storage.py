# -*- coding: utf-8 -*-
"""爬虫落盘层：全模块唯一的写文件动作 + 去重 / 断点辅助。

自 ``pipeline.py`` 拆出。只依赖 ``schema``，不感知任何数据源细节，
因此可以单独阅读与复用。
"""
import json
import os
import re

from .schema import (PATHS, doc_path, load_lines, record_failure, save_json,
                     validate)


# ======================= 落盘：全模块唯一的写文件动作 =======================
def _write_json_line(path, obj):
    """往 jsonl 追加一行（目录不存在时自动建）"""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _nmpa_failed_ids():
    """已在失败清单里的药名集合（用于去重，重跑不会重复追加）"""
    ids = set()
    for ln in load_lines(PATHS["nmpa_failed"]):
        try:
            ids.add(json.loads(ln)["keyword"])
        except Exception:  # noqa: BLE001
            continue
    return ids


def record_nmpa_failure(keyword, reason):
    """记录抓不到批准文号的药名。同药名只记一次。

    这些药名以前会变成「全空记录」混进 drugs.json（实测 26 条），下游分不清是
    抓失败还是确实没有。现在单独记一份清单，drugs.json 里只留有据可查的记录。
    """
    if not keyword or keyword in _nmpa_failed_ids():
        return
    _write_json_line(PATHS["nmpa_failed"], {"keyword": keyword, "reason": reason})


def sink_doc(out_dir, doc):
    """落盘一个统一 schema 文档，返回 'skip'（已存在）/ 'ok' / 'fail'。

    断点续爬与失败记录都在这里收口，爬虫本身不碰文件系统。
    """
    path = doc_path(out_dir, doc["doc_id"])
    if os.path.exists(path):
        return "skip"
    missing = validate(doc)
    if missing:
        record_failure(out_dir, doc["doc_id"], "缺字段:" + ",".join(missing))
        return "fail"
    save_json(path, doc)
    return "ok"


def _count_docs(out_dir):
    """统计目录内有效文档数（排除失败记录）"""
    if not os.path.isdir(out_dir):
        return 0
    return len([f for f in os.listdir(out_dir) if f.endswith(".json")])


def otc_done(out_dir):
    """已抓过的公告 id 集合（按子目录里是否有 index.json 判断）。"""
    done = set()
    if not os.path.isdir(out_dir):
        return done
    for name in os.listdir(out_dir):
        sub = os.path.join(out_dir, name)
        if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "index.json")):
            done.add(name)
    return done


def _safe_filename(name):
    """把附件名清洗成合法文件名（保留中文，去掉路径分隔与非法字符）。"""
    name = (name or "").strip().replace("/", "_").replace("\\", "_")
    name = re.sub(r'[:*?"<>|]', "_", name).strip(". ")
    return name or None
