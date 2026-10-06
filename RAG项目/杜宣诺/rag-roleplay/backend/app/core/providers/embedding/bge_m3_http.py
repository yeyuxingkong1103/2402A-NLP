import httpx  # 导入 httpx，用于异步 HTTP 请求

from ....config import get_settings  # 从上层 config 模块导入配置获取函数


class BgeM3Http:  # 定义 BGE-M3 的 HTTP 调用封装类
    def __init__(self, base_url: str | None = None):  # 构造函数，可传入 base_url
        self.base_url = (base_url or get_settings().embedding_base_url).rstrip("/")  # 优先用传入值，否则用配置，并去掉尾部斜杠

    async def _post(self, payload: dict) -> dict:  # 定义内部异步 POST 方法
        async with httpx.AsyncClient(timeout=60) as client:  # 创建带 60 秒超时的异步 HTTP 客户端
            r = await client.post(f"{self.base_url}/embed", json=payload)  # 向 /embed 发 POST 请求
            r.raise_for_status()  # 状态码非 2xx 时抛异常
            return r.json()  # 返回解析后的 JSON

    async def encode_dense(self, texts: list[str]) -> list[list[float]]:  # 批量编码稠密向量
        texts = [t for t in texts if t and t.strip()]  # 过滤掉空字符串/纯空白字符串
        if not texts:  # 如果没有有效文本
            return []  # 直接返回空列表
        data = await self._post({"texts": texts})  # 调用 /embed 接口
        return data["dense"]  # 返回稠密向量列表

    async def encode_sparse(self, texts: list[str]) -> list[dict[int, float]]:  # 批量编码稀疏向量
        texts = [t for t in texts if t and t.strip()]  # 过滤掉空字符串/纯空白字符串
        if not texts:  # 如果没有有效文本
            return []  # 直接返回空列表
        data = await self._post({"texts": texts})  # 调用 /embed 接口
        return [{int(k): v for k, v in s.items()} for s in data["sparse"]]  # 把稀疏向量的键转成 int，值保持 float

    async def encode_query(self, text: str) -> tuple[list[float], dict[int, float]]:  # 编码单条查询
        if not text or not text.strip():  # 如果查询为空或纯空白
            return [], {}  # 返回空的稠密和稀疏向量
        return (await self.encode_dense([text]))[0], (await self.encode_sparse([text]))[0]  # 分别取稠密和稀疏的第一条结果