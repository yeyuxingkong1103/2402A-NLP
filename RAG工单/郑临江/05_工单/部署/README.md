# 部署：多轮 Query 理解

从工单根目录准备独立 Python 环境，安装研发依赖，再检查目录与语法：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r "研发/requirements.txt"
python run.py --check
python run.py 研发/app.py
```

附件路径在 `../研发/config.py` 配置，模型 API 参数继续通过环境变量提供。
脚本型工单使用本地/服务器 Python 环境执行；模型训练型工单需准备 GPU、模型与数据集。
上线前按 `../测试/README.md` 完成真实输入验证，并保存日志与可回退版本。
