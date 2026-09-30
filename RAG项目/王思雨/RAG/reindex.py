# -*- coding: utf-8 -*-
"""一键重建模块（第 13 步新增）：把「解析 → 增强 → 入库」串成一条链路。

**三处共用这一份实现**，避免同一套编排写三遍、改三处：
  - 命令行：`python -m reindex` 或 `python -m ingest --reindex`
  - 脚本  ：`bash scripts/reindex.sh`
  - 接口  ：`POST /kb/rebuild`（后台任务）

三步各自的产物：

| 步骤 | 做什么 | 输入 | 输出 |
|---|---|---|---|
| 1. 解析 | `ingest.parse_all_pdfs()` | `.env` 的 `PDF_FILES` | `data/parsed_chunks.json` |
| 2. 增强 | `enrich.enrich_all()` | parsed | `data/enriched_chunks.json` |
| 3. 入库 | `vector_store.insert_chunks(recreate=True)` | enriched | Milvus 集合 |

**注意**：每一步都会**覆盖**上一轮的产物（含 Milvus 集合会被删掉重建），
这是一键重建的预期行为——它的语义就是"按当前配置从零重来"。
"""

import sys                                  # 读命令行参数、设置退出码
import time                                 # 统计每步耗时

from logger import get_logger               # 日志工具

logger = get_logger("reindex")              # 创建本模块的 logger 实例

# 三步的名字，供接口返回、日志与打印共用；顺序即执行顺序
STEPS = ("ingest", "enrich", "index")

# 三步的中文名，打印时用
STEP_LABELS = {"ingest": "1. 解析 PDF", "enrich": "2. 离线增强", "index": "3. 向量入库"}


def _warm_up_torch() -> None:
    r"""解析之前先给 torch 与 BGE-m3 热身，再让后面可能上场的 PaddleOCR 加载。

    **为什么必须热身（实测出来的）**：Windows 上 PaddleOCR 与 torch 的本地库有 DLL 冲突——
    先加载 PaddleOCR、之后才 import torch 时，会报
    `[WinError 127] Error loading "torch\lib\shm.dll" or one of its dependencies`，
    向量化直接失败。顺序反过来（先 torch 后 PaddleOCR）实测两边都能正常工作。

    扫整个 PDF_DIR 时目录里可能含真扫描件，解析会触发 PaddleOCR；
    一旦 OCR 先加载，第 3 步入库就再也加载不了 torch。
    所以这里在解析**之前**先把 torch 与 BGE-m3 拉起来。
    BGE 模型是单例，这里加载过，第 3 步直接复用，不额外花时间（约 2 秒）。
    """
    try:                                                            # 预热失败不该中断解析与增强
        import vector_store                                         # 延迟导入：会加载 BGE-m3
        vector_store.embed_texts(["预热"])                          # 触发 torch 与模型加载
        logger.info("torch 与 BGE-m3 已预热（避免 PaddleOCR 抢占 DLL）")   # 记录一次
    except Exception as exc:                                        # 预热失败
        logger.warning("torch/BGE 预热失败，入库步骤可能会失败：%s", exc)   # 只告警，继续


def _fail(detail: dict, done: list, step: str, error) -> dict:
    """统一的失败返回：记日志并组装结果字典，供调用方判断与打印。"""
    logger.error("一键重建在第 %d 步（%s）失败，后续步骤不再执行：%s",
                 len(done) + 1, step, error)                      # 记下失败在哪一步
    return {"ok": False, "steps": list(done), "failed": step,    # 已完成步骤与失败步骤
            "detail": detail, "error": str(error)}               # 明细与错误信息


def run_reindex() -> dict:
    """依次执行解析、增强、入库三步；任一步失败即停止，不继续后面的步骤。

    **额外加了一道保护**：解析出 0 个块时**不往下走**。
    因为第 3 步是 `recreate=True`（先删集合再建），
    万一 `PDF_FILES` 全指到了不存在的文件，就会拿空数据重建，把好的索引清成 0 条。
    宁可在这里停住报错，也不要静默清库。

    :return: ``{"ok", "steps", "failed", "detail", "error"}``
             ``detail[步骤名]`` 里是 ``{"in", "out", "elapsed"}``
    """
    detail = {}                                                    # 每步的输入数/输出数/耗时
    done = []                                                      # 已完成的步骤名
    _warm_up_torch()                                               # 先热身 torch，免得被 OCR 抢走 DLL

    # ---------- 第 1 步：按 PDF_FILES 解析 PDF → parsed_chunks.json ----------
    import ingest                                                  # 延迟导入：解析链路依赖较重
    start = time.time()                                            # 计时起点
    try:                                                           # 解析或落盘失败都要兜住
        results = ingest.parse_all_pdfs()                          # 按 .env 的清单解析（空则扫全目录）
        ingest.save_chunks_to_json(results)                        # 结果写入 parsed_chunks.json
    except Exception as exc:                                       # 本步失败
        return _fail(detail, done, "ingest", exc)                  # 停在这里，返回失败
    chunks = sum(len(r["chunks"]) for r in results)                # 解析出的语义块总数
    detail["ingest"] = {"in": len(results), "out": chunks,         # 输入 PDF 数、输出块数
                        "elapsed": round(time.time() - start, 1)}  # 本步耗时
    done.append("ingest")                                          # 标记本步完成
    logger.info("第 1/3 步 解析完成：输入 %d 个 PDF → 输出 %d 块，耗时 %.1f 秒",
                len(results), chunks, detail["ingest"]["elapsed"])   # 打印本步结果
    if chunks <= 0:                                                # 一个块都没解析出来
        return _fail(detail, done, "ingest",                        # 立即中止，避免清空索引
                     "解析出 0 个块，已中止以免把 Milvus 索引清成空；"
                     "请检查 .env 的 PDF_FILES 与 PDF_DIR 是否配对")

    # ---------- 第 2 步：离线增强 → enriched_chunks.json ----------
    import enrich                                                  # 延迟导入
    start = time.time()                                            # 计时起点
    try:                                                           # 增强失败要兜住
        stats = enrich.enrich_all()                                # 低质过滤 → 去重 → 摘要 → 父子块
    except Exception as exc:                                       # 本步失败
        return _fail(detail, done, "enrich", exc)                  # 停在这里，返回失败
    final = int(stats.get("final", 0)) if isinstance(stats, dict) else 0   # 增强后的最终块数
    detail["enrich"] = {"in": chunks, "out": final,                # 输入块数、输出块数
                        "elapsed": round(time.time() - start, 1)}  # 本步耗时
    done.append("enrich")                                          # 标记本步完成
    logger.info("第 2/3 步 增强完成：输入 %d 块 → 输出 %d 块，耗时 %.1f 秒",
                chunks, final, detail["enrich"]["elapsed"])        # 打印本步结果

    # ---------- 第 3 步：向量化并重建 Milvus 集合 ----------
    import vector_store                                            # 延迟导入：会加载 BGE-m3
    start = time.time()                                            # 计时起点
    try:                                                           # 入库失败要兜住
        items = vector_store.load_chunks_from_json()               # 读增强结果（已自动跳过父块）
        inserted = vector_store.insert_chunks(items, recreate=True)   # 先删集合再建并写入
    except Exception as exc:                                       # 本步失败
        return _fail(detail, done, "index", exc)                   # 停在这里，返回失败
    detail["index"] = {"in": len(items), "out": inserted,          # 输入子块数、写入条数
                       "elapsed": round(time.time() - start, 1)}   # 本步耗时
    done.append("index")                                           # 标记本步完成
    logger.info("第 3/3 步 入库完成：输入 %d 条子块 → 写入 %d 条，耗时 %.1f 秒",
                len(items), inserted, detail["index"]["elapsed"])   # 打印本步结果

    return {"ok": True, "steps": done, "failed": None,             # 三步全过
            "detail": detail, "error": ""}                          # 无失败步骤与错误


def print_report(result: dict) -> None:
    """把三步的输入数/输出数/耗时与最终 row_count 打成一张表。"""
    print("=" * 78)                                                 # 分隔线
    print("一键重建结果（解析 → 增强 → 入库）")                        # 标题
    print("=" * 78)                                                 # 分隔线
    for name in STEPS:                                              # 按固定顺序打印三步
        info = result["detail"].get(name)                           # 取该步明细
        label = STEP_LABELS[name]                                   # 该步中文名
        if not info:                                                # 该步没跑
            print("  %-14s 未执行" % label)                          # 打印未执行
            continue                                                # 继续下一行
        print("  %-14s 输入 %5d → 输出 %5d，耗时 %7.1f 秒"
              % (label, info["in"], info["out"], info["elapsed"]))   # 打印本步三个数
    print("-" * 78)                                                 # 分隔线
    if result["ok"]:                                                # 三步全过
        _print_row_count(result["detail"]["index"]["out"])          # 传写入条数，等计数追平
        print("  结论：一键重建成功")                                  # 成功结论
    else:                                                           # 有步骤失败
        print("  失败于步骤：%s" % result["failed"])                   # 打印失败步骤
        print("  错误：%s" % result["error"])                        # 打印错误原因
    print("=" * 78)                                                 # 分隔线


def _print_row_count(expected: int = 0) -> None:
    """查 Milvus 的最终条数并打印；查不到只提示，不算失败。

    **必须把期望条数传进来**：Milvus 是最终一致的，写入后计数不会立刻追平。
    `verify_insert` 会按 `expected` 轮询等待；传 0 时它立即返回——
    实测因此读到过偏小的中间值（写入 724 却读到 704，几十秒后才追平）。
    """
    try:                                                            # Milvus 可能不可用
        import vector_store                                         # 延迟导入
        info = vector_store.verify_insert(expected=expected)        # 带上期望值，等计数追平再读
        print("  Milvus 集合 %s 最终 row_count = %d"
              % (info["collection"], info["row_count"]))            # 打印最终条数
    except Exception as exc:                                        # 查询失败
        print("  （查询 Milvus 条数失败：%s）" % exc)                  # 只提示，不影响结论


def main() -> None:
    """命令行入口：`python -m reindex`。任一步失败返回非 0 退出码。"""
    logger.info("开始一键重建：解析 → 增强 → 入库")                    # 记启动日志
    result = run_reindex()                                          # 执行三步
    print_report(result)                                            # 打印结果表
    if not result["ok"]:                                            # 有失败
        sys.exit(1)                                                 # 返回非 0，供脚本判断


if __name__ == "__main__":                                          # 支持 python -m reindex
    main()                                                          # 执行命令行入口
