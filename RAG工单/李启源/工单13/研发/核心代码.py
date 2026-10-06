"""工单13：RAG性能瓶颈识别与优化的可运行核心代码。"""
from time import perf_counter, sleep

def run():
    timer = {}
    for name, seconds in (("query", 0.01), ("retrieve", 0.02), ("generate", 0.03)):
        start = perf_counter()
        sleep(seconds)
        timer[name] = round(perf_counter() - start, 4)
    return timer

if __name__ == "__main__":
    print(run())
