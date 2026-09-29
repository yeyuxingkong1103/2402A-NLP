# -*- coding: utf-8 -*-
"""旧版 v6 恢复运维脚本已从精简版移除。

当前项目通过 backend/pipeline.py 统一执行 PDF 解析、清洗、分块、向量化和入库。

在链路中的位置：
    空壳模块 —— 老版本里这是一个"从 Milvus 备份恢复知识库"的运维脚本。
    精简版取消了单独的恢复流程，改由 backend/pipeline.py 的 build_pipeline()
    从原始 PDF 重建（原始 PDF 一直保留在 data/pdfs/，本身就是最可靠的备份）。

    保留本文件的原因同 backend/health.py：让按老路径 import 的代码得到
    明确的迁移指引，而不是一个难懂的 ImportError。
"""


def main() -> int:
    """打印迁移提示并返回成功码。

    返回：
        0（成功）。返回 0 而不是非 0，是因为这不是"执行失败"，
        而是"该功能已按设计移除"—— 用非 0 会让 CI/定时任务误报错误。
    """
    print("restore_v6.py 已从精简版移除；请使用 /api/upload 或 build_pipeline() 重建文档。")
    return 0


if __name__ == "__main__":
    # 保留命令行入口，便于运维同学直接执行时能看到迁移提示
    raise SystemExit(main())
