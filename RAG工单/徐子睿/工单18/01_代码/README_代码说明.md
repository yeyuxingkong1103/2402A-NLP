# 工单 18 · 代码说明

## 目录

```
01_代码/
├─ document_quality_assessment/        # Skill 本体（资源层）
│  ├─ SKILL.md                         # 指令层（含元数据 frontmatter）
│  ├─ assessment_config.yaml           # 全部可调参数（阈值不硬编码）
│  ├─ config.py                        # 配置加载（yaml → dataclass）
│  ├─ detectors.py                     # 五大功能实现
│  ├─ skill.py                         # 主入口 assess()（进度回调/中断恢复）
│  ├─ report.py                        # JSON + HTML 简报
│  ├─ workflow.py                      # document_ingestion_workflow 路由
│  ├─ api.py                           # POST /v1/document/quality-inspection
│  ├─ cli.py                           # 命令行入口
│  └─ __init__.py
└─ tests/
   └─ test_skill.py                    # 7 项单元测试
```

## 快速开始

```bash
cd 01_代码
set PYTHONPATH=%CD%                    # Windows；Linux 用 export PYTHONPATH=$PWD

python -m unittest discover -s tests -v
python -m document_quality_assessment.cli --root <文件夹> --out report.json --html report.html
```

## 依赖

- 必需：`pymupdf`（PDF 文本/页数）
- 可选：`pyyaml`（读 yaml 配置，缺省用内置默认）、`docx`（读 Word）、`fastapi`+`uvicorn`（API）

## 集成到 RAGFlow

1. 把 `document_quality_assessment/` 放入 RAGFlow 技能目录，Agent 决策节点挂载（读取 `SKILL.md`）；
2. 在 RAGFlow API 注册 `api.build_app()` 的路由（或复用 `run_inspection()`）；
3. 工作流用 `workflow.document_ingestion_workflow()` 做「质检 → 按标签路由解析器」。
