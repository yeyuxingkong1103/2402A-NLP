# 专利 PDF 检索优化过程记录

---

## 1. 原因分析（初始失败）

PDF 为图片型且分辨率低，导致检索失败，原因分析如下：

| 原因 | 说明 |
| --- | --- |
| OCR 识别率低 | PDF 为图片型且分辨率低，文字粘连 |
| 分块导致信息丢失 | Paper 分块导致专利文本被切碎，关键信息丢失 |
| 相似度阈值过高 | 召回相关块太少 |
| 未使用 Rerank | 排序不准确 |

---

## 2. 第二轮优化：提高 PDF 解析质量

**措施**：用 PyMuPDF 将 PDF 每页以 300 DPI 渲染为 PNG，再上传解析。

```python
import fitz

doc = fitz.open("CN100342976C.pdf")
for page in doc:
    pix = page.get_pixmap(dpi=300)
    pix.save(f"page_{page.number}.png")
```

**效果**：OCR 准确率提升，问题 1、2 能召回正确块，问题 3-6 仍失败。

---

## 3. 第三轮优化：调整分块策略

**措施**：分块方法改为 One 或调整 `chunk_token_num` 为 512，确保图纸说明完整。

**效果**：问题 3-6 的文本描述被完整保留，但图像信息仍未关联。

---

## 4. 第四轮优化：启用 Rerank 并调整权重

**措施**：启用 bge-reranker-v2-m3，相似度阈值 0.1，向量权重 0.5，关键词权重 0.5。

**效果**：问题 1、2、6 正确，问题 3-5 仍错误。

---

## 5. 第五轮优化：多模态信息融合

**措施**：对图纸区域单独 OCR，将标注文字与图片说明合并到同一分块。

**代码修改（pdf_parser.py）：**

```python
def _extract_figure(self, image, page_num):
    ocr_result = self.ocr(image)
    text = " ".join([item[1] for item in ocr_result])
    return {"image": image, "text": text, "page": page_num}

def chunk(self, text, images):
    for img in images:
        if img['text']:
            text += "\n[图片标注] " + img['text']
    # 分块逻辑...
```

**效果**：6 个问题全部正确，准确率 100%。

---

## 6. 最终配置

| 配置项 | 值 |
| --- | --- |
| 解析方法 | Paper（使用高清预处理图片） |
| 分块 token 数 | 512 |
| 相似度阈值 | 0.1 |
| 向量权重 | 0.5 |
| 关键词权重 | 0.5 |
| Rerank 模型 | bge-reranker-v2-m3 |
| 额外处理 | 图纸区域单独 OCR 并合并上下文 |
