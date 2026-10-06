# 演示截图清单与拍摄说明

> 工单编号：**人工智能NLP-RAG-Query理解优化任务**
> 本目录存放演示视频录制时使用的截图（建议 PNG，1920×1080）。
> 截图由运行本工单脚本/界面产生，属于「运行后生成」的素材，请勿手工修图篡改内容。

---

## 一、命名规范

```
序号-场景-要点.png
例如：01-建索引-两份招股说明书入库.png
      06-指代消解可视化-第4轮改写.png
```

## 二、必拍截图清单

| 序号 | 文件名（建议） | 画面内容 | 获取方式 | 要突出的要点 |
|------|----------------|----------|----------|--------------|
| 01 | `01-index-two-docs.png` | 建索引输出：两份文档、块数、向量数、BM25 文档数 | `python build_index.py` | 跨文档知识库已就绪 |
| 02 | `02-query-understanding-test.png` | 单测逐用例通过列表 | `python query_understanding.py --offline` | 24 条离线断言全通过 |
| 03 | `03-core-assertion.png` | 核心断言详情（第 4 轮改写前后对照） | `results/understanding_demo.md` 的「核心断言详解」 | 「那 X 呢？」→「X 的法定代表人是谁？」且 0 次 LLM 调用 |
| 04 | `04-ui-overview.png` | 界面全貌：左侧对话、右侧指代消解面板 | `python chat_ui.py` | 布局清晰、信息分区明确 |
| 05 | `05-ui-turn2-turn3.png` | 第 2/3 轮对话与改写结果 | 在界面依次提问 | 「他」「这个公司」被消解为兴图新科 |
| 06 | `06-ui-turn4-rewrite.png` | **重点**：第 4 轮右侧面板「检索式（已改写）」高亮 | 提问「那武汉力源信息技术股份有限公司呢？」 | 省略句补全 + 主体切换，跨到《招股说明书2》 |
| 07 | `07-ui-citations-and-latency.png` | 引用来源卡片 + 性能卡片（本轮耗时、阶段明细） | 同一次提问后滚动右侧面板 | 来源页码可溯源、耗时透明 |
| 08 | `08-ui-turn5-image.png` | 第 5 轮组织结构图问题：命中 image 类型片段 | 提问第 5 轮 | 图像块参与检索，答案含销售处清单 |
| 09 | `09-ui-feedback.png` | 点赞/点踩按钮与反馈状态 | 点击「有帮助」 | 反馈写回 AdaptiveReranker |
| 10 | `10-ablation-compare.png` | 消融对比表（关闭 vs 开启）+ 核心证据 | `python ablation_multiturn.py --fast` | 关闭时第 2/3/4 轮检索失败/准确率下降 |
| 11 | `11-evaluation-summary.png` | 评估指标汇总与逐轮明细 | `python run_evaluation.py` | 准确率、RAGAS 指标、3 秒达标率 |
| 12 | `12-conversation-md.png` | 5 轮对话记录 Markdown（含阶段耗时） | `results/multi_turn_conversation.md` | 完整链路留痕 |

## 三、拍摄要点

1. **窗口整洁**：只保留必要窗口；终端字号建议 ≥ 16pt，确保视频压缩后可读。
2. **敏感信息**：不要拍到 API Key、内网地址、个人目录外的隐私信息。
3. **指代消解是主角**：第 4 轮的改写面板至少给 2 张图（改写前提问 / 改写后面板），
   并在图上用箭头或高亮框标注「原始问题 → 检索式」。
4. **对比才有说服力**：消融实验截图务必同时包含两组的「检索式」与「判定」列。
5. **页码可核对**：引用来源截图尽量包含页码，便于评审对照 PDF 复核。
6. **原始分辨率**：截图保留原始分辨率，不要二次压缩；引用到文档时再缩放。

## 四、复现步骤（三步）

```bash
cd 工单05-Query理解优化-多轮对话/src
python build_index.py                 # 截图 01
python query_understanding.py         # 截图 02、03
python chat_ui.py                     # 截图 04 ~ 09（按脚本逐轮提问）
python ablation_multiturn.py --fast   # 截图 10
python run_evaluation.py              # 截图 11
# 截图 12：打开 results/multi_turn_conversation.md
```

## 五、目录约定

- 本目录只放截图与说明，不放视频（视频按提交要求单独归档）；
- 截图对应的原始结果文件在 `results/` 下，可作为「结果与截图一致」的交叉验证。
