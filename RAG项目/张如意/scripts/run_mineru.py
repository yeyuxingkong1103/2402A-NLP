# -*- coding: utf-8 -*-
"""对本期 3 份国标跑 MinerU（本地版），产物落到 data/external/mineru/。

用法：
    python scripts/run_mineru.py                 # 跑黄瓜/辣椒/大蒜三份
    python scripts/run_mineru.py 黄瓜            # 只跑某一份

为什么要单独一个脚本：
  - MinerU 装在独立 venv（D:/mineru/.venv-mineru），不是本项目的解释器
  - 必须**全量跑**（不加 -s/-e）：带页码范围时 page_idx 是相对偏移，
    与 PDF 物理页对不上，交叉比对会整体错位
  - 产物放 data/external/ 而不是 data/processed/：CLAUDE.md 定义前者
    「不进语料统计、不进索引」，MinerU 输出是对照物不是解析产物

跑完接着跑：python scripts/diff_mineru.py

【模块说明（补充）】
  输入：data/raw/ 下三份国标 PDF（见 PDFS 映射）。
  输出：data/external/mineru/{作物}/ 下的 MinerU 解析产物（由 MinerU 自行写出）。
  退出码：0=全部成功，1=有 PDF 缺失或解析失败，2=MinerU 解释器缺失/未知作物名。
  ⚠️ 阅读提示：本文件代码与文档注释存在不一致处（如 MINERU_PY 的反斜杠路径、
    run() 内局部变量命名），以实际代码为准；单独运行可能抛 NameError。
"""
import os
import subprocess
import sys
import time

MINERU_PY = r"D:\mineru\.venv-mineru\Scripts\python.exe"   # MinerU 独立虚拟环境里的 python 解释器
RAW = os.path.join("data", "raw")                          # 原始 PDF 目录
OUT = os.path.join("data", "external", "mineru")           # MinerU 产物目录（external=对照物，不进索引）

PDFS = {
    "黄瓜": "GB_Z 26581-2011 黄瓜生产技术规范.pdf",
    "辣椒": "GB_Z 26583-2011 辣椒生产技术规范.pdf",
    "大蒜": "GB_Z 26578-2011 大蒜生产技术规范.pdf",
}


def run(which):
    """依次对 which 列表里的作物跑 MinerU。

    参数 which：作物名列表（必须是 PDFS 的键）。
    返回：进程退出码——0=全部成功；1=有 PDF 缺失或某份解析失败。
    """
    if not os.path.exists(MINERU_PY):
        print(f"找不到 MinerU 解释器：{MINERU_PY}")
        print("（MinerU 装在独立 venv 里，不在本项目的 Python 环境）")
        return 2                                        # 解释器不存在 -> 退出码 2
    os.makedirs(OUT, exist_ok=True)
    rc = 0                                              # 汇总退出码：任何一份失败即置 1
    for crop in which:
        pdf = os.path.join(RAW, PDFS[crop])             # 该作物的原始 PDF 路径
        if not os.path.exists(pdf):
            print(f"[{crop}] 缺 PDF：{pdf}")
            rc = 1
            continue                                    # 缺文件只记录，不中断后续作物
        print(f"\n{'='*70}\n[{crop}] {PDFS[crop]}\n{'='*70}", flush=True)
        t0 = time.time()
        # 注意：不加 -s/-e，全量跑，保证 page_idx 与 PDF 物理页一一对应
        cmd = [MINERU_PY, "-m", "mineru.cli.client",
               "-p", pdf, "-o", os.path.join(OUT, crop)]  # -p 输入 PDF；-o 输出目录（按作物分子目录）
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")            # 捕获输出；解码失败用替换符兜底，不让子进程异常带崩
        dt = time.time() - t0
        tail = "\n".join((p.stdout or "").strip().splitlines()[-6:])   # 只回显子进程 stdout 最后 6 行
        print(tail)
        if p.returncode != 0:
            print(f"[{crop}] 失败 rc={p.returncode}")
            print("\n".join((p.stderr or "").strip().splitlines()[-10:]))  # 失败时回显 stderr 最后 10 行
            rc = 1
        else:
            print(f"[{crop}] 完成，用时 {dt/60:.1f} 分钟")   # 用时按分钟显示，保留 1 位小数
    return rc


if __name__ == "__main__":
    args = sys.argv[1:]
    which = args if args else list(PDFS)                # 不带参数 = 默认跑全部三份
    bad = [w for w in which if w not in PDFS]           # 校验作物名是否合法
    if bad:
        print(f"未知作物 {bad}，可选：{list(PDFS)}")
        sys.exit(2)
    print(f"MinerU: {MINERU_PY}")
    print(f"输出到: {OUT}")
    sys.exit(run(which))
