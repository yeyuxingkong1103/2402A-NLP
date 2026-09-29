# RAGAS 评测

`dataset.json` 是最小示例集。先执行 `pip install -r requirements-eval.txt`，再启动系统、上传示例 PDF，并配置评测用模型。运行 `python evaluation/run_ragas.py`，脚本会登录、调用真实聊天接口，再计算指标。

RAGAS 的 faithfulness 与 answer_relevancy 需要评判 LLM/embedding。默认复用 `DEEPSEEK_API_KEY` 和本地 BGE 配置；评测本身会产生 DeepSeek API 费用。生产评测至少准备 50 条经医疗专业人员审核的问题、标准答案与来源上下文，并按版本保存结果。
