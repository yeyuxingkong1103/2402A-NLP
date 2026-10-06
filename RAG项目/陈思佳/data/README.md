# data 目录说明

本目录用于本地运行时存放知识库资料和处理产物，不建议提交真实资料或生成结果到 Git。

常见子目录：

- `raw/`：原始 PDF、Word、Markdown、TXT 资料
- `parsed/`：MinerU 或本地解析后的 Markdown
- `chunks/`：向量化前的 JSONL 分块文件
- `mineru/`：MinerU 返回的 ZIP、图片、表格等完整产物

如果需要复现知识库，请将资料放入 `data/raw/` 后按 README 中的命令重新解析、分块和入库。
