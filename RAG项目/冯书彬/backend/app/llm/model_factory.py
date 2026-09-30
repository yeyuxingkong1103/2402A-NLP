from backend.app.llm.deepseek_client import DeepSeekClient


def create_deepseek_client(api_key: str, base_url: str = "https://api.deepseek.com", model: str = "deepseek-chat") -> DeepSeekClient:
    # 工厂只负责创建 DeepSeek 客户端，不读取环境变量或编排聊天流程。
    return DeepSeekClient(api_key=api_key, base_url=base_url, model=model)
