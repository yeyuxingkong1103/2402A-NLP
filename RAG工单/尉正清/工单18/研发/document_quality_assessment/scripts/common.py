# 工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流
"""公用部分：读配置、抽正文、断点恢复

## 为什么正文抽取要单独抽出来

五大功能里有三个（长度分布、重复检测、敏感信息）都要读文档正文，而
**读正文是整个流程里最慢的一步**（1700 份文档，多数是扫描件）。
所以只读一遍，把结果缓存到 workdir，后面的检查都复用 —— 这也是「支持中断恢复」
的基础：缓存落盘了，断了重跑不用从头来。
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
SKILL_DIR = HERE.parent
DEFAULT_CONFIG = SKILL_DIR / "assessment_config.yaml"


def load_config(path=None):
    """读 YAML 配置。所有阈值都从这里来，代码里不写死。"""
    p = Path(path) if path else DEFAULT_CONFIG
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    # 补默认值，避免调用方到处写 .get(..., 默认)
    cfg.setdefault("scan", {})
    cfg["scan"].setdefault("workers", 0)
    cfg["scan"].setdefault("skip_dirs", [])
    cfg["scan"].setdefault("max_parse_mb", 200)
    return cfg


# ---------------------------------------------------------------- 遍历

def iter_files(root, cfg):
    """遍历目录，产出文件路径。

    跳过隐藏目录和配置里点名的目录（`.git`、`node_modules` 这类），
    否则知识库里混进代码仓库会得到一堆没意义的统计。
    """
    root = Path(root)
    if root.is_file():
        yield root
        return
    exts = {e.lower() for e in cfg["scan"]["extensions"]}
    skip = set(cfg["scan"]["skip_dirs"])
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith(".")]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in exts:
                yield p


# ---------------------------------------------------------------- 断点恢复

class Checkpoint:
    """把每个文件的解析结果攒起来，每 N 个落一次盘。

    工单备注 3：「处理 1700 份文档可能耗时较长，请考虑进度反馈和支持中断恢复」。
    缓存键用「路径 + 大小 + 修改时间」，文件改过就重算。
    """

    def __init__(self, workdir, every=100):
        self.dir = Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "parsed.jsonl"
        self.every = max(1, every)
        self.cache = {}
        self._buf = []
        self._load()

    @staticmethod
    def key(path):
        p = Path(path)
        try:
            st = p.stat()
            return f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}"
        except OSError:
            return str(path)

    def _load(self):
        if not self.file.exists():
            return
        for line in self.file.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                self.cache[r["key"]] = r
            except Exception:
                continue

    def get(self, path):
        return self.cache.get(self.key(path))

    def put(self, path, record):
        record = {"key": self.key(path), **record}
        self.cache[record["key"]] = record
        self._buf.append(record)
        if len(self._buf) >= self.every:
            self.flush()

    def flush(self):
        if not self._buf:
            return
        with self.file.open("a", encoding="utf-8") as f:
            for r in self._buf:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self._buf = []

    def reset(self):
        """丢掉缓存，强制重新解析。

        什么时候需要：改了**抽取类**配置（敏感信息正则、SimHash 取多少字符）。
        只改**判定类**阈值（扫描页占比、长度区间、标签规则）不需要 ——
        那些是出报告时才算的，缓存的是事实不是结论。
        """
        self.cache.clear()
        self._buf = []
        if self.file.exists():
            self.file.unlink()


# ---------------------------------------------------------------- 杂项

def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human(n):
    for unit in ("", "K", "M", "G"):
        if abs(n) < 1000:
            return f"{n:.0f}{unit}"
        n /= 1000
    return f"{n:.0f}T"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def utf8_console():
    """让 Windows 控制台能打 UTF-8。Linux 上本来就是 UTF-8，这是空操作。

    两种乱码都实际踩到了：
    1. 控制台按 GBK 解码，中文标签和路径打成问号；
    2. 文件名里混着非 GBK 字节（知识库常见的下载乱码名）时，
       直接抛 UnicodeEncodeError 把整个质检中断掉。
    所以入口脚本第一件事就是调这个。
    """
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
