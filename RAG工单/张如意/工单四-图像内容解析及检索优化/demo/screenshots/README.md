# 演示截图目录

**工单编号：人工智能NLP-RAG-图像内容解析及检索优化**

本目录存放演示视频拍摄过程中的关键截图与录屏素材，与 `demo/演示脚本.md` 的分镜一一对应。

## 命名规范

```
分镜号_内容_日期.png
例如：03_image_extract_clip_20260206.png
```

## 建议截图清单

| 文件名建议 | 对应分镜 | 画面要求 |
|------------|----------|----------|
| `02_figure_page_probe_YYYYMMDD.png` | 分镜2 | 终端输出 `p38/p39/p72` 的绘图对象数、内嵌图数、文本长度，**能看清"23 张内嵌图全是水印"这一点** |
| `03_image_extract_cli_YYYYMMDD.png` | 分镜3 | `python src/image_extractor.py --docs 2` 的运行输出，含 `page_render` 与 CLIP 标签 |
| `03_org_chart_render_YYYYMMDD.png` | 分镜3 | `results/images/招股说明书2/p39_fullpage.png` 放大后能看到"销售部 → 大客户销售部 → 6 个销售处" |
| `03_ic_chart_embedded_YYYYMMDD.png` | 分镜3 | 第 72 页抽出的饼图与条形图，**能看到 `14.0%` 与 `-2.0%`** |
| `04_image_descriptions_YYYYMMDD.png` | 分镜4 | `results/image_descriptions.md`，含"销售部的直接下级（4 个）"与"最大/最小值/负值类别" |
| `05_ablation_before_YYYYMMDD.png` | 分镜5 | 对照组 `--collection wo04_text` 输出"未能找到该问题的答案"、命中图像 0 个 |
| `05_ablation_after_YYYYMMDD.png` | 分镜5 | 实验组输出命中图像块 + 层级事实 + 完整答案 + 精确度 |
| `06_ablation_report_YYYYMMDD.png` | 分镜6 | `results/image_ablation.md` 的总体对比表（准确率提升那一行） |
| `06_evaluation_report_YYYYMMDD.png` | 分镜6 | `results/evaluation_multimodal.md` 顶部指标行（准确率、耗时） |
| `07_web_image_card_YYYYMMDD.png` | 分镜7 | Web 界面点击「图像问题5」后的答案 + 图像卡片 |
| `07_web_english_ask_YYYYMMDD.png` | 分镜7 | 英文提问的问答效果（多语言验收） |

## 截图规范

1. **分辨率**：1920×1080，使用 125% 以上缩放保证中文与数字清晰；
2. **标注**：用红框圈出关键信息（如 `-2.0%`、"4 个"、"6 个"、耗时数字），
   每条截图旁边标注一行小字说明"看什么"；
3. **隐私**：截图前清屏，避免暴露 `DEEPSEEK_API_KEY` / `RAG_VLM_API_KEY` 等密钥；
4. **一致性**：所有截图使用同一终端主题（建议浅色背景，投影更清晰）。

## 与视频的关系

* 视频成片建议放在 `demo/演示视频.mp4`（或提交网盘链接并在本文件登记）；
* 截图作为视频的补充证据，验收时可快速翻阅；
* 若重跑脚本导致结果数字变化，请同步更新截图并在文件名中更新日期。

## 登记表

| 文件 | 说明 | 拍摄日期 | 拍摄人 |
|------|------|----------|--------|
| （待填） | | | |
