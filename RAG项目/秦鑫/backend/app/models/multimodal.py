from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import httpx

from ..config import Settings
from .llm import require_siliconflow_key

MEDIA_PROMPT = """
你正在为法律 RAG 系统抽取用户上传材料。你的任务不是回答法律问题，而是把材料中**可直接识别**的事实尽量完整、结构化地写出来，便于后续检索和引用。

硬性要求：
1. 只写材料里能看见、能听见、能直接读出的内容，不要推理，不要补故事，不要下法律结论。
2. 不要省略关键字段。金额、时间、地点、主体、动作、对话、文件名、备注、备注中的时间点、截图状态，都要尽量提取。
3. 遇到看不清、听不清、缺上下文、缺原件、缺完整聊天或流水时，必须明确写“未识别/看不清/缺少上下文”，不要跳过。
4. 如果同一张截图里有多段对话、多个金额或多个时间点，要分开列，不要合并成一句空泛总结。
5. 如果内容很少，也要如实写出“实际识别到什么”，不要输出模板化空话。

请按下面固定结构输出，标题不要改：

【材料类型】
截图 / 聊天记录 / 合同 / 收据票据 / 转账记录 / 录音 / 视频 / 照片 / 证件 / 其他

【可识别原文】
尽量逐行保留原文，聊天按说话顺序保留；合同、票据、转账截图保留关键字段原样；看不清处写“看不清”。

【主体信息】
出现的人名、公司/单位、账号、手机号、平台昵称、签名、盖章、收款方、付款方、聊天双方身份线索。

【时间与金额】
日期、时间、期限、工资、借款、转账、赔偿、租金、押金、价款等，保留原始数字和单位。

【时间轴】
如果材料里出现明确时间码、日期、聊天时间、录音顺序、视频片段顺序，按顺序列出来。
优先写“绝对时间”；没有绝对时间时，写“第1段 / 第2段 / 第3段”这类相对顺序，并说明这是片段顺序，不是精确时间。
每一段尽量包含“时间点 + 发生了什么 + 谁对谁做了什么”。

【关键事实线索】
谁对谁说了什么、谁向谁转了什么、谁要求了什么、谁拒绝了什么、是否有承诺、催告、解除、辞退、报警、就医、交付、收款、付款、约定服务内容等。

【证据状态】
是否截图/复印件/节选/转述，是否缺少上下文、原件、完整聊天、完整录音时长、完整流水、拍摄时间、来源账号。

【待核实点】
列出仍然无法确认、但会影响判断的关键信息。

音频请先转写主要对话，再按上述结构提取；视频请同时描述画面文字、可听语音、人物行为、关键事件和时间轴。不要输出法律建议，不要输出案件结论。
""".strip()


class MultimodalAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def analyze_media(self, path: Path, media_type: str, filename: str) -> str:
        require_siliconflow_key(self.settings)
        data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        content = [{"type": "text", "text": MEDIA_PROMPT}, {"type": f"{media_type}_url", f"{media_type}_url": {"url": f"data:{mime};base64,{data}"}}]
        response = httpx.post(f"{self.settings.siliconflow_base_url.rstrip('/')}/chat/completions", headers={"Authorization": f"Bearer {self.settings.siliconflow_api_key}"}, json={"model": self.settings.multimodal_model, "messages": [{"role": "user", "content": content}], "temperature": 0.1, "max_tokens": self.settings.multimodal_max_tokens}, timeout=self.settings.multimodal_timeout)
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"] or "").strip()
