"""S5 向量化的实现模块。对外唯一入口是 backend/embed_chunks.py。

模块划分：model 编码口径（可复用）/ store 读写产物。
"""

RULE_VERSION = "embed-v1+bge-m3"

# 本机已有完整权重。HF 缓存里那份是不完整的（只有 22 MB 的 config/tokenizer）。
MODEL_DIR = r"E:\资料\BAAI--bge-m3"
WEIGHT_FILE = "pytorch_model.bin"

# D1 裁决：4096。实测最长 chunk 为 3260 token，此值之下零截断。
MAX_LENGTH = 4096
BATCH_SIZE = 16
DIM = 1024

NORM_TOLERANCE = 1e-5

EXIT_OK, EXIT_FAIL, EXIT_BAD_INPUT = 0, 1, 2
EXIT_MODEL, EXIT_EMPTY_TEXT, EXIT_BAD_VECTOR = 3, 4, 5


class EmbedError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message
