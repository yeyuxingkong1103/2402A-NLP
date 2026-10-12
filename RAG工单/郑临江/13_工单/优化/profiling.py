# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
应用级性能分析：cProfile 采集（Python 语言级分析器）+ snakeviz 可视化。
"""

# 五阶段目录兼容：从测试/优化入口加载本工单研发模块。
import sys as _sys
from pathlib import Path as _Path
_project_root = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_project_root / "研发"))
_sys.path.append(str(_project_root / "优化"))
import cProfile
import io
import pstats


def profile_func(fn, *args, **kwargs):
    """对 fn 进行 cProfile 分析，输出慢函数排行并保存 profile 文件。"""
    pr = cProfile.Profile()
    pr.enable()
    result = fn(*args, **kwargs)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(20)
    print("===== cProfile 慢函数 Top20 =====")
    print(s.getvalue())
    pr.dump_stats("profile.prof")
    print("[profiling] 已保存 profile.prof，可用 `snakeviz profile.prof` 可视化查看")
    return result


def snakeviz_hint():
    """提示如何在浏览器中可视化 profile 文件。"""
    print("安装：pip install snakeviz")
    print("运行：snakeviz profile.prof")
