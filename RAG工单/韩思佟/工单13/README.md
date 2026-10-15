# 工单13 · RAG 性能瓶颈识别与优化

工单编号：人工智能NLP-RAG-RAG 性能瓶颈识别与优化

## 一、工单目标

分阶段计时 + cProfile 定位瓶颈并优化，每次提问检索返回 ≤3s，给出前后对比。

## 二、怎么跑

```bash
python scripts/w13_perf_profile.py
```

## 三、产出物
- `outputs/w13/工单13_性能优化报告.md`
- `outputs/w13/cProfile剖析.txt`
- `outputs/w13/优化前_耗时.json`
- `outputs/w13/优化后_耗时.json`

## 五、需要截图的地方

见同目录 `截图清单.md`。