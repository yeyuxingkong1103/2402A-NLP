"""提示词与角色预设的统一出口。

两部分：role_presets 是纯数据（内置角色的人设表：心理咨询师、律师），templates 是纯逻辑（把检索资料 +
短期记忆 + 提问拼成一条发给 LLM 的 messages）。两者都不依赖 pipeline / store，可以单独 import。

链路位置：pipeline.answer / answer_stream 用 build_messages 拼提示词、用 ROLE_PRESETS 登记
角色；main.py 启动时按 ROLE_PRESETS 批量注册角色；scripts/seed.py 用同一份预设灌知识库。

本文件只是转发层：仓库内的调用方目前都直接 import 子模块（`from .prompt.templates import
build_messages` 这种），这里留一个包级入口给外部调用与测试用；以后要挪动子模块文件名，
在这里改一次转发即可，不必挨个改调用方。
"""
from .role_presets import ROLE_PRESETS
from .templates import build_messages

__all__ = ["ROLE_PRESETS", "build_messages"]
