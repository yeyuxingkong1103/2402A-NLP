"""pytest 公共夹具：离线 dummy 配置 + 假 Milvus/Retriever，测试不依赖 Ollama / Milvus Lite。

夹具是分层的，越往上替掉的真东西越多，按需取用：

    dummy_settings   纯配置（llm/embed=dummy、SQL=临时 sqlite、记忆=进程内）
    fake_pipeline    真 RAGPipeline + 假 milvus/retriever —— 测编排，不测检索质量
    client           HTTP 层：真 lifespan 跑完再换掉 app.state.pipeline

「离线」由两件事保证：一是 llm_provider / embed_provider 选 dummy（app/core/llm.py 与
embedding.py 的 dummy 分支不发起任何网络请求，答复是回显用户问题、向量是确定性哈希）；
二是下面两个假件替掉 Milvus 与整条检索链路。所以这一层断言的是管线编排与接口协议，
模型答得好不好、向量召回准不准不在这里管（要验真实链路走 test_integration.py）。

假件故意做得极简、返回写死的文本（「限制钠盐摄入」），是为了让断言能直接比对具体字符串：
检索结果有没有进提示词、有没有回给前端，只有靠这种可辨认的字面量才抓得住。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 必须在导入任何 app.* 之前设置。
#
# logging_config 是模块级单例（_CONFIGURED 守卫），导入 app.main 时它就按**真实配置**
# 打开了 logs/app.log 的文件 handler；之后再改 settings.log_dir 不会生效。后果不只是
# 脏：日志里分不清哪条来自测试、哪条是真实故障——排查时会被自己写的测试误导。
#
# load_dotenv 默认不覆盖已有环境变量，所以这里设的值优先于 .env。
_TMP = Path(tempfile.mkdtemp(prefix="rag-tests-"))
os.environ["LOG_DIR"] = str(_TMP / "logs")

# 存储与记忆后端也钉死到临时目录（2026-09-21 审计发现：以前只钉了 LOG_DIR，
# 而 `with TestClient(app)` 会跑**真实 lifespan**——于是每次 pytest 都按 .env 连上
# 生产 MySQL/Redis，执行 create_all、ALTER TABLE、预设角色 upsert，测试直接写生产库；
# 组件一重启，全部接口用例在 setup 阶段 ERROR（假红）。
os.environ["APP_ENV"] = ""  # 别让 .env.{APP_ENV} 分层参与测试
os.environ["SQL_URL"] = f"sqlite:///{_TMP / 'app.db'}"
os.environ["MILVUS_DB_URI"] = str(_TMP / "milvus.db")
os.environ["MEMORY_BACKEND"] = "memory"  # 不依赖 Redis；多轮记忆用 conftest 的假件
# 刻意**不**钉 LLM/Embedding provider：test_integration.py 需要真 Ollama（自带 skip 守卫），
# 而 lifespan 不会在启动期调用它们（都是懒连接），钉了反而会把集成测试悄悄降级成 dummy。

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.core.embedding import EmbeddingClient  # noqa: E402
from app.core.llm import LLMClient  # noqa: E402
from app.core.pipeline import RAGPipeline  # noqa: E402
from app.core.reranker import create_reranker  # noqa: E402
from app.core.store.memory import InMemoryMemoryStore  # noqa: E402
from app.core.store.sql_store import SQLStore  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture
def dummy_settings(tmp_path) -> Settings:
    """全离线配置：LLM/Embedding 用 dummy，SQL 用临时 sqlite。"""
    return Settings(
        llm_provider="dummy",
        embed_provider="dummy",
        milvus_uri=str(tmp_path / "milvus.db"),
        sql_url=f"sqlite:///{tmp_path / 'app.db'}",
        memory_backend="memory",
        memory_max_turns=10,
        top_k=3,
        score_threshold=0.0,
        reranker="score_fusion",
        log_dir=str(tmp_path / "logs"),
    )


class FakeMilvus:
    """内存假 Milvus，返回一条固定命中。"""

    def __init__(self):
        self.rows = []

    def ping(self) -> bool:
        return True

    def search(self, role_id, qv, top_k):
        return [
            {"id": "d1", "score": 0.9, "text": "限制钠盐摄入", "title": "指南", "source": "s", "chunk_index": 0}
        ]

    def list_texts(self, role_id):
        return [{"id": "d1", "text": "限制钠盐摄入", "title": "指南", "source": "s", "chunk_index": 0}]

    def insert(self, role_id, chunks):
        self.rows.extend(chunks)

    def delete_by_source(self, role_id, source):
        return 0

    def count(self):
        return len(self.rows)


class FakeRetriever:
    """替掉整条检索链路：混合召回、BM25、精排、query 改写全都不参与。

    需要验这些环节的用例（如 test_pipeline.py 里管池宽那两条）自己造 retriever/reranker，
    不走这个夹具——否则会把「池宽没生效」这类接线错误一并假掉。
    """

    def retrieve(self, query, role_id, top_k=None):
        return [
            {"id": "d1", "score": 0.9, "text": "限制钠盐摄入", "title": "指南", "source": "s", "chunk_index": 0}
        ]

    def invalidate(self, role_id=None):
        pass


@pytest.fixture
def make_fake_pipeline(dummy_settings):
    """按需组装假管线；传入 llm 可替换默认的 dummy LLM。

    用于覆盖「特定 provider 行为」的场景——例如某些 openai_compat 中转会把
    推理内联在 content 里（而 ollama 走独立字段，真实模型复现不出来）。
    """

    def _make(llm=None) -> RAGPipeline:
        return RAGPipeline(
            settings=dummy_settings,
            llm=llm or LLMClient(dummy_settings),
            embedding=EmbeddingClient(dummy_settings),
            milvus=FakeMilvus(),
            sql=SQLStore(dummy_settings),
            memory=InMemoryMemoryStore(dummy_settings.memory_max_turns),
            retriever=FakeRetriever(),
            reranker=create_reranker(dummy_settings),
        )

    return _make


@pytest.fixture
def fake_pipeline(make_fake_pipeline) -> RAGPipeline:
    """真实 LLM(dummy)/Embedding(dummy)/SQL，假 Milvus/Retriever 组成的管线。"""
    return make_fake_pipeline()


@pytest.fixture
def client(fake_pipeline):
    """HTTP 接口测试客户端：用假管线替换 lifespan 组装出的真实管线。"""
    with TestClient(app) as c:
        app.state.pipeline = fake_pipeline
        yield c
