# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-18-实现文档质量评估Skill并集成至智能体工作流
# 模块职责：加载与校验 assessment_config.yaml，提供带默认值兜底的点号取值接口，
#           保证所有核心判断阈值都来自配置而非硬编码。

import os                                                          # 路径处理
import copy                                                        # 深拷贝默认配置

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))             # Skill 包目录
WORK_ORDER_DIR = os.path.dirname(SKILL_DIR)                        # 工单十八根目录
DEFAULT_CONFIG_FILE = os.path.join(WORK_ORDER_DIR, "assessment_config.yaml")   # 默认配置

# 配置缺失时的兜底默认值：保证配置被删减后 Skill 仍可运行
FALLBACK = {
    "paths": {"default_target": "", "output_dir": "data",
              "state_file": "data/assessment_state.json"},
    "scan": {"extensions": [".pdf", ".docx", ".md", ".txt"], "recursive": True},
    "pdf_triage": {"char_threshold": 30, "scan_ratio_threshold": 0.70,
                   "text_ratio_threshold": 0.30, "borderline_margin": 0.10,
                   "max_pages_probe": 0, "text_extract_flags": "text"},
    "length": {"percentiles": [25, 50, 75, 90, 99],
               "bins": [0, 200, 1000, 3000, 10000, 50000, 1000000000],
               "bin_labels": ["<200", "200-1k", "1k-3k", "3k-10k", "10k-50k", ">50k"]},
    "dedup": {"md5_enabled": True, "simhash_enabled": True, "simhash_bits": 64,
              "hamming_threshold": 3, "simhash_min_chars": 200,
              "band_count": 4, "max_context_chars": 120},
    "sensitive": {"enabled_types": {"mobile": True, "email": True,
                                    "id_card": True, "bank_card": False},
                  "context_chars": 30, "max_hits_per_type": 200,
                  "exclude_patterns": []},
    "classify": {"primary_labels": [], "quality_tags": [],
                 "too_short_chars": 200, "empty_text_chars": 10},
    "routing": {"label_to_parser": {}, "review_first_tags": ["Corrupted"],
                "ocr_max_workers": 2},
    "runtime": {"resume": True, "progress_every": 100, "html_report": True,
                "json_indent": 2, "error_sample_limit": 50},
}


def _merge(base, override):
    """把 override 递归合并进 base 的副本（dict 逐键合并，其他类型整体覆盖）。"""
    out = copy.deepcopy(base)                                      # 先复制兜底默认值
    for key, val in (override or {}).items():                      # 逐键覆盖
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], val)                       # 字典递归合并
        else:
            out[key] = val                                         # 其他类型直接覆盖
    return out


class Config(object):
    """配置对象：支持 cfg.get("pdf_triage.char_threshold", 30) 形式的点号取值。"""

    def __init__(self, data, source):
        self._data = data                                          # 合并后的配置字典
        self.source = source                                       # 配置来源文件（便于报告溯源）

    def get(self, path, default=None):
        """按 "a.b.c" 路径取值；缺失时返回 default。"""
        node = self._data                                          # 从根开始
        for part in path.split("."):                               # 逐级下钻
            if not isinstance(node, dict) or part not in node:
                return default                                     # 任一级缺失即兜底
            node = node[part]
        return node

    def section(self, name):
        """取整个配置段（返回浅拷贝的 dict）。"""
        val = self._data.get(name)
        return dict(val) if isinstance(val, dict) else {}

    def as_dict(self):
        """返回完整配置字典（报告中记录本次运行的口径）。"""
        return copy.deepcopy(self._data)

    def resolve_path(self, path_value, base=None):
        """把配置中的相对路径解析为绝对路径；已是绝对路径则原样返回。"""
        if not path_value:
            return path_value
        if os.path.isabs(path_value):
            return path_value
        return os.path.normpath(os.path.join(base or WORK_ORDER_DIR, path_value))


def load_config(config_file=None):
    """加载配置文件并合并兜底默认值；文件不存在时全部走默认值并给出提示。"""
    path = config_file or DEFAULT_CONFIG_FILE                      # 配置文件位置
    raw, note = {}, ""                                             # 原始配置与提示
    if os.path.exists(path):                                       # 文件存在则解析
        try:
            import yaml                                            # 延迟导入
            with open(path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        except Exception as exc:                                   # 配置损坏也不阻断运行
            note = "配置文件解析失败（%s），已全部使用内置默认值" % exc
    else:
        note = "未找到配置文件 %s，已全部使用内置默认值" % path
    return Config(_merge(FALLBACK, raw), path), note
