"""数据包文件哈希（全项目唯一实现）。

content_hash 是增量判定（unchanged / new / updated）的依据，
文件哈希算法必须只有一处实现，避免多副本将来分叉导致判定错乱。
"""

from hashlib import sha256
from pathlib import Path


def file_sha256(file_path: Path) -> str:
    return "sha256:" + sha256(file_path.read_bytes()).hexdigest()
