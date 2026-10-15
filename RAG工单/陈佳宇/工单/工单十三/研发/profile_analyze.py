# -*- coding: utf-8 -*-
# profile_analyze.py cProfile性能分析入口
import cProfile
import pstats
from rag_perf_main import rag_full_pipeline

def run_profile():
    prof_file = "perf_stat.prof"
    query = "武汉力源本次募集资金投资哪些项目？"
    cProfile.runctx(
        "rag_full_pipeline(query)",
        globals(),
        locals(),
        filename=prof_file
    )
    # 打印top20耗时函数到控制台
    stats = pstats.Stats(prof_file)
    stats.sort_stats(pstats.SortKey.TIME)
    print("=====cProfile top耗时函数=====")
    stats.print_stats(20)

if __name__ == "__main__":
    run_profile()
