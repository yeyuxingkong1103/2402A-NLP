"""官方 LightRAG 后端入口。需要 OPENAI_API_KEY，首次执行建立独立工作目录。"""
import argparse, asyncio, os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--docs", nargs="+", required=True)
parser.add_argument("--query", required=True)
args = parser.parse_args()
if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("请设置 OPENAI_API_KEY 后再运行官方 LightRAG。")
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
    from lightrag.utils import EmbeddingFunc
except ImportError as exc:
    raise SystemExit("请先执行：python -m pip install -r requirements-lightrag.txt") from exc
from pypdf import PdfReader

async def run():
    root = Path(__file__).parent
    rag = LightRAG(working_dir=str(root / "outputs" / "official_lightrag"),
                   llm_model_func=gpt_4o_mini_complete,
                   embedding_func=EmbeddingFunc(embedding_dim=1536, max_token_size=8192, func=openai_embed))
    await rag.initialize_storages()
    for raw in args.docs:
        path = Path(raw)
        text = "\n".join((page.extract_text() or "") for page in PdfReader(path).pages)
        await rag.ainsert(text)
    print(await rag.aquery(args.query, param=QueryParam(mode="hybrid")))

asyncio.run(run())
