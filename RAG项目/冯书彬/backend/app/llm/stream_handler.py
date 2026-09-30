from collections.abc import AsyncIterator


async def collect_stream(stream: AsyncIterator[str]) -> list[str]:
    # 安全收集器只在内存中收集已校验流片段，不负责前端推送，也不降低 DeepSeek 客户端的完整验证边界。
    chunks: list[str] = []
    async for chunk in stream:
        chunks.append(chunk)
    return chunks
