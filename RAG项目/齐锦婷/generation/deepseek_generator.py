import httpx

from backend.app.core.config import get_settings


settings = get_settings()
DISCLAIMER = "\n\n免责声明：以上内容仅供学习参考，不构成正式法律意见；如涉及真实案件，请咨询专业律师。"


class LLMService:
    """负责调用 DeepSeek 生成最终法律回答。"""

    async def answer(self, question: str, references: list[dict], short_memory: list[dict], long_memory: list[str]) -> str:
        # 如果没有配置 DeepSeek 密钥，就返回基于依据的兜底回答，方便本地演示。
        if not settings.deepseek_api_key or settings.deepseek_api_key.startswith("请在这里"):
            return self._fallback_answer(question, references)
        # 构造引用材料，要求模型只能基于这些材料回答。
        reference_text = self._format_references(references)
        # 构造短期记忆文本，帮助模型理解当前会话上下文。
        short_memory_text = "\n".join([f"{item['role']}：{item['content']}" for item in short_memory])
        # 构造长期记忆文本，体现用户级历史偏好。
        long_memory_text = "\n".join(long_memory)
        # 系统提示词明确法律问答边界、回答风格和引用格式。
        system_prompt = (
            "你是一个面向普通用户的中国法律 RAG 咨询助手。"
            "你必须优先依据提供的参考资料回答，不能编造不存在的法条、机构、流程或结论。"
            "你的目标不是只复述法条，而是把检索到的法律依据转化为清晰、可执行的建议。"
            "回答必须使用中文，并采用 Markdown 排版。"
            "回答结构固定为："
            "1. 先用一句话给出核心结论；"
            "2. 分条说明相关法律依据，每个关键结论后必须用【依据1】这样的格式标注来源；"
            "3. 给出用户现在可以怎么做的具体步骤，按紧急程度排序；"
            "4. 给出需要准备或保留的证据材料；"
            "5. 提醒可能的风险、限制或资料不足之处；"
            "6. 最后给出2到3个用户可以继续追问的问题。"
            "如果问题涉及人身安全、家暴、伤害、威胁等紧急场景，要先提示立即报警或寻求紧急保护。"
            "如果参考资料不足以支持某项建议，要明确写出‘现有资料不足以判断’，不要强行下结论。"
        )
        # 用户提示词包含问题、记忆和检索依据。
        user_prompt = f"用户问题：{question}\n\n短期记忆：\n{short_memory_text}\n\n长期记忆：\n{long_memory_text}\n\n参考资料：\n{reference_text}"
        # DeepSeek 官方接口兼容 OpenAI chat/completions。
        url = f"{settings.deepseek_base_url.rstrip('/')}/v1/chat/completions"
        # 构造请求头。
        headers = {"Authorization": f"Bearer {settings.deepseek_api_key}", "Content-Type": "application/json"}
        # 构造请求体。
        payload = {
            "model": settings.deepseek_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }
        # 设置请求超时时间。
        timeout = httpx.Timeout(settings.deepseek_timeout_seconds)
        # 异步调用 DeepSeek。
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
        # 如果 API 返回错误，抛出异常给上层处理。
        response.raise_for_status()
        # 解析模型返回内容。
        data = response.json()
        # 取第一条回答。
        answer = data["choices"][0]["message"]["content"].strip()
        # 确保固定免责声明存在。
        return answer + DISCLAIMER

    def _format_references(self, references: list[dict]) -> str:
        # 把检索结果格式化成【依据N】列表。
        lines: list[str] = []
        # 逐条引用材料编号。
        for index, item in enumerate(references, start=1):
            # 拼接文件名、页码、标题路径和正文。
            lines.append(
                f"【依据{index}】文件：{item.get('filename', '未知文档')}；页码：{item.get('page_number', 0)}；"
                f"标题：{item.get('title_path', '')}\n{item.get('content', '')}"
            )
        # 返回完整引用文本。
        return "\n\n".join(lines)

    def _fallback_answer(self, question: str, references: list[dict]) -> str:
        # 没有检索依据时，明确告诉用户资料不足。
        if not references:
            return f"没有检索到足够的法律依据，暂时无法回答：{question}{DISCLAIMER}"
        # 没有大模型时，用前几条依据生成结构化兜底答案。
        reference_lines = []
        for index, item in enumerate(references[:3], start=1):
            content = item.get("content", "").strip()
            reference_lines.append(f"{index}. {content[:260]}【依据{index}】")
        return (
            f"## 核心结论\n\n"
            f"已根据问题“{question}”检索到相关法律依据。当前未启用大模型生成服务，下面先给出基于检索结果的基础建议。\n\n"
            f"## 相关法律依据\n\n"
            + "\n\n".join(reference_lines)
            + "\n\n## 你现在可以怎么做\n\n"
            "1. 先确认自己的具体事实是否符合上述法条中的条件。\n"
            "2. 保存合同、通知、聊天记录、付款记录、照片、录音录像等原始证据。\n"
            "3. 如果涉及紧急人身或财产风险，优先报警、联系主管机关或寻求律师帮助。\n"
            "4. 如果只是一般争议，可以先整理时间线和证据，再考虑协商、投诉、仲裁或诉讼。\n\n"
            "## 需要继续确认\n\n"
            "- 事情发生的时间、地点和对方身份是什么？\n"
            "- 你手里有哪些书面或电子证据？\n"
            "- 你希望达成的目标是制止行为、赔偿、解除合同，还是确认权利？"
            f"{DISCLAIMER}"
        )
