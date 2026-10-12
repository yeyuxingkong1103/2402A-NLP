# 工单编号：人工智能NLP-RAG-混合检索任务
# 文件说明：MinerU 解析链路的停机实测。全量入库走的是 data/parsed 缓存，缓存存在时并不会真的调用 MinerU；
#           本脚本从招股说明书2 切出 SLICE_PAGES 页实跑一次 MinerU 子进程，证明"调用 MinerU"这条代码路径确实可用。
#           跑完自动清理临时产物，不污染 data/parsed。
import shutil, sys, time                                                           # 标准库
from pathlib import Path                                                           # 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))                    # 让脚本能 import app 包
from app import config as C                                                        # 配置
from app import ingest as I                                                        # 被验证的入库模块

SLICE_PAGES = 3                                                                    # 切片页数：够触发解析，又不至于等太久
SRC_PDF = "招股说明书2.pdf"                                                         # 用较小的一份做实测


def main() -> int:                                                                 # 返回退出码
    """切页 → 实跑 MinerU → 校验产物 → 清理。任一环节失败返回 1，绝不静默通过。"""
    src = C.RAW_DIR / SRC_PDF                                                      # 源 PDF
    if not src.exists():                                                           # 源文件缺失
        print(f"FAIL  找不到 {src}")                                                # 明确报因
        return 1                                                                   # 退出
    tmp = C.ROOT / "data" / "_mineru_smoke"                                        # 临时工作目录
    if tmp.exists():                                                               # 有上次残留
        shutil.rmtree(tmp)                                                         # 先清干净
    tmp.mkdir(parents=True)                                                        # 建目录

    import fitz                                                                    # pymupdf，用来切页
    doc = fitz.open(src)                                                           # 打开源文件
    cut = fitz.open()                                                              # 新文档
    cut.insert_pdf(doc, from_page=10, to_page=10 + SLICE_PAGES - 1)                # 取第 11 页起的若干页
    slice_pdf = tmp / "smoke_slice.pdf"                                            # 切片文件
    cut.save(slice_pdf)                                                            # 落盘
    cut.close(); doc.close()                                                       # 关闭句柄

    saved = C.PARSED_DIR                                                           # 备份原产物目录
    C.PARSED_DIR = tmp                                                             # 把入库模块的输出目录临时指向临时区
    try:                                                                           # 保证异常也能还原配置
        t0 = time.time()                                                           # 计时
        print(f"[实测] 对 {SLICE_PAGES} 页切片实跑 MinerU（这一步真的会调子进程）…", flush=True)
        ok = I.run_mineru(slice_pdf)                                               # 实跑 MinerU
        el = time.time() - t0                                                      # 耗时
        print(f"[实测] run_mineru 返回 {ok}，耗时 {el:.1f}s")                        # 打印
        if not ok:                                                                 # 调用失败
            print("FAIL  MinerU 子进程未能产出 content_list，解析路径不可用")          # 明确报因
            return 1                                                               # 退出
        blocks = I.mineru_blocks(slice_pdf)                                        # 读产物并转块
        print(f"[实测] 产物解析出 {len(blocks)} 块")                                 # 打印
        for b in blocks[:3]:                                                       # 抽样展示
            print(f"        P{b['page_no']} {b['block_type']} :: {b['text'][:60]!r}")  # 页码/类型/文本
        if not blocks:                                                             # 产物为空
            print("FAIL  产物为空，解析路径返回空结果")                                 # 明确报因
            return 1                                                               # 退出
        print("PASS  MinerU 解析路径实测可用")                                        # 通过
        return 0                                                                   # 退出
    except Exception as exc:                                                       # 任何异常
        print(f"FAIL  {type(exc).__name__}: {exc}")                                 # 明确报因
        return 1                                                                   # 退出
    finally:                                                                       # 无论成败
        C.PARSED_DIR = saved                                                       # 还原输出目录
        shutil.rmtree(tmp, ignore_errors=True)                                     # 清理临时产物，不留垃圾
        print("[实测] 临时产物已清理")                                                # 提示


if __name__ == "__main__":                                                         # 允许直接运行
    sys.exit(main())                                                               # 退出码供 CI 使用
