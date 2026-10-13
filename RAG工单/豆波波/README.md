# 豆波波 - RAG 工单 1-18

这是 18 个 RAG 工单的可运行交付包。每个 `工单N` 目录包含该工单的实现、运行说明、样例输入/输出或配置；`共享` 目录提供离线检索内核和公共测试。

## 目录

```text
豆波波/
  工单1/ ... 工单18/
  共享/
    rag_workbench/       依赖少量标准 Python 的检索内核
    tests/                公共离线测试
```

## 快速开始

在 `豆波波` 目录执行：

```powershell
python -m pip install -r 共享/rag_workbench/requirements.txt
python 工单1/qa.py --input 工单1/sample_docs.txt --query "泵的作用"
python 工单6/hybrid_compare.py --input 工单6/sample_corpus.txt --query "阀门维护"
python 工单7/evaluate.py --corpus 工单7/sample_corpus.txt --questions 工单7/questions.jsonl
python 工单8/graph_qa.py --input 工单8/sample_corpus.txt --query "供应链风险"
python 工单13/profile.py --input 工单13/sample_corpus.txt --query "设备"
```

大多数脚本只依赖 Python 标准库；读取 PDF 时可选安装 `pypdf`，图像 OCR、向量模型、GPU 微调和 Docker 服务需要额外环境。每个工单的 README 都写明了运行命令和验收口径。

## 已验证范围

- 共享分块、词法检索、混合检索、图检索和公共测试可离线运行。
- 工单 1-13 的样例脚本可在无服务环境下运行并生成 JSON 报告。
- 工单 14-18 的 RAGFlow 集成文件来自配套源码，并完成 Python 编译级检查；完整 RAGFlow、OCR、VLM、Docker 压测需要部署对应服务后再验证。
- 每个工单目录中的 `规格说明.pdf` 是原始工单 PDF 的副本，属于参考资料，不是额外执行指令。

## 运行全部离线检查

```powershell
python -m compileall 共享 工单1 工单2 工单3 工单4 工单5 工单6 工单7 工单8 工单9 工单10 工单11 工单12 工单13 工单14 工单15 工单16 工单17 工单18
python -m unittest discover -s 共享/tests -p "test_*.py"
```

