"""路径解析。

输入：data/parsed/{doc_id}/<文档名>/auto/<文档名>_content_list.json
输出：data/clean/{doc_id}.blocks.jsonl
      data/clean/{doc_id}.dropped.jsonl
"""

from __future__ import annotations

import os

from .errors import CleanError, EXIT_BAD_INPUT

CONTENT_LIST_SUFFIX = "_content_list.json"
V2_SUFFIX = "_content_list_v2.json"


def project_root() -> str:
    """backend/clean/paths.py -> backend/clean -> backend -> 仓库根。"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def default_input_root(root: str) -> str:
    return os.path.join(root, "data", "parsed")


def default_output_dir(root: str) -> str:
    return os.path.join(root, "data", "clean")


def discover_doc_ids(input_root: str) -> list[str]:
    """列出 data/parsed/ 下的文档标识（目录名即 doc_id）。"""
    if not os.path.isdir(input_root):
        raise CleanError(
            EXIT_BAD_INPUT,
            "输入根目录不存在：%s\n（期望 data/parsed/ 下有已解析的文档目录）" % input_root,
        )
    return sorted(
        name
        for name in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, name))
    )


def find_content_list(doc_dir: str) -> str:
    """在一份文档的产物目录里定位 content_list.json。

    可能有多个（多次解析），此时不静默挑选，直接报错要求人工用
    --content-list 指定。
    """
    hits: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(doc_dir):
        for fn in filenames:
            if fn.endswith(V2_SUFFIX):
                continue
            if fn.endswith(CONTENT_LIST_SUFFIX):
                hits.append(os.path.join(dirpath, fn))
    hits.sort()

    if not hits:
        raise CleanError(
            EXIT_BAD_INPUT,
            "在 %s 下找不到 *_content_list.json\n（上游 S2 解析可能未完成）" % doc_dir,
        )
    if len(hits) > 1:
        raise CleanError(
            EXIT_BAD_INPUT,
            "在 %s 下找到 %d 份 content_list.json，无法自动选择，请用 --content-list 指定：\n%s"
            % (doc_dir, len(hits), "\n".join("  " + h for h in hits)),
        )
    return hits[0]


def doc_file_name(content_list_path: str) -> str:
    """由产物文件名反推源文件名：<名>_content_list.json -> <名>.pdf"""
    base = os.path.basename(content_list_path)
    stem = base[: -len(CONTENT_LIST_SUFFIX)]
    return stem + ".pdf"
