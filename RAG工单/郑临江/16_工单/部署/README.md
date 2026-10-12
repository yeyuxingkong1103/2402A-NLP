# 部署：视觉语言模型微调

从工单根目录准备独立 Python 环境，安装研发依赖，再检查目录与语法：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r "研发/requirements.txt"
python run.py --check
python run.py 研发/data_convert.py
```

附件路径在 `../研发/config.py` 配置，模型 API 参数继续通过环境变量提供。
脚本型工单使用本地/服务器 Python 环境执行；模型训练型工单需准备 GPU、模型与数据集。
上线前按 `../测试/README.md` 完成真实输入验证，并保存日志与可回退版本。

## LoRA 训练

在 WSL/Linux 环境中先按设计说明注册 imdr_vlm 数据集，然后从工单根目录运行 `bash 部署/launch.sh`。脚本会切换到研发目录，使用其中的 train_config.yaml；训练配置与 Python 虚拟环境必须一致。
