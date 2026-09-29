"""问答落库的模型名（批次 31）。

背景：此前 `persist_turn` 落库时 `model` 恒为字面量 None（旧注释"LLM 客户端未暴露
模型名"），结果 `chat_messages.model` 全为 NULL——回答是哪个模型生成的无法回溯。
本批改为从**当轮生效配置**（settings.llm_model）读取；历史 NULL 行不回填、不猜测。

本文件锁两件事：
1. 落库调用里传的 model 等于当轮配置值（不是写死的 None）；
2. 配置缺失/为空时降级为 None 且不抛错（禁止伪造）。
"""

# SimpleNamespace：构造轻量配置替身，避免碰真实 Settings 的必填校验
from types import SimpleNamespace

# 配置模块：monkeypatch 目标是"调用点所在的模块命名空间"
import app.core.config as config_module
# 被测对象：落库编排入口与模型名读取函数
from app.api.chat_persistence import current_llm_model, persist_turn


class RecordingChatStore:
    """记录落库调用的存储替身（只实现 persist_turn 依赖的方法）。"""

    def __init__(self) -> None:
        # 每次调用的完整参数，供断言检查 model 取值
        self.calls: list[dict] = []

    def append_message_pair(
        self,
        session_key: str,
        user_id: str,
        question: str,
        answer: str,
        sources: list[dict],
        model: str | None = None,
        message_id: str | None = None,
    ) -> None:
        """与真实实现同签名，原样记录参数。"""
        self.calls.append(
            {
                "session_key": session_key,
                "question": question,
                "answer": answer,
                "sources": sources,
                "model": model,
                "message_id": message_id,
            }
        )


def _run_persist(store: RecordingChatStore) -> None:
    """跑一次一问一答落库（不接 Redis，只验落库入参）。"""
    persist_turn(
        chat_store=store,
        short_term_memory=None,
        user_id="user-1",
        session_key="session-1",
        question="经济补偿怎么算",
        # 结果对象只需 answer / sources 两个字段（persist_turn 只读这两个）
        result=SimpleNamespace(answer="按工作年限每满一年支付一个月工资[1]", sources=[]),
        request_id="test-request",
        message_id="message-1",
    )


# ---------------------------------------------------------------- 落库模型名


def test_persist_turn_writes_model_from_current_config(monkeypatch) -> None:
    """落库的 model 等于当轮配置的模型名（此前恒为 None）。"""
    monkeypatch.setattr(
        config_module, "settings", SimpleNamespace(llm_model="deepseek-flash-test")
    )
    store = RecordingChatStore()

    _run_persist(store)

    assert len(store.calls) == 1
    assert store.calls[0]["model"] == "deepseek-flash-test"
    # 落库字段必须真有着落，不能是 None（本批修复的核心）
    assert store.calls[0]["model"] is not None


def test_persist_turn_model_follows_config_change(monkeypatch) -> None:
    """换配置就换落库值——锁住"取自配置"而非任何写死的字符串。"""
    store = RecordingChatStore()

    monkeypatch.setattr(config_module, "settings", SimpleNamespace(llm_model="model-a"))
    _run_persist(store)
    monkeypatch.setattr(config_module, "settings", SimpleNamespace(llm_model="model-b"))
    _run_persist(store)

    assert [call["model"] for call in store.calls] == ["model-a", "model-b"]


def test_persist_turn_keeps_other_fields_untouched(monkeypatch) -> None:
    """只改 model 取值：其余落库参数逐字不变（question/answer/message_id/sources）。"""
    monkeypatch.setattr(
        config_module, "settings", SimpleNamespace(llm_model="deepseek-flash-test")
    )
    store = RecordingChatStore()

    _run_persist(store)

    call = store.calls[0]
    assert call["session_key"] == "session-1"
    assert call["question"] == "经济补偿怎么算"
    assert call["answer"] == "按工作年限每满一年支付一个月工资[1]"
    assert call["sources"] == []
    assert call["message_id"] == "message-1"


# ---------------------------------------------------------------- 降级：禁止伪造


def test_current_llm_model_returns_none_when_unconfigured(monkeypatch) -> None:
    """配置为空串或读不到时一律返回 None（宁缺勿造）。"""
    # 空串：配置存在但未填值，不算有效模型名
    monkeypatch.setattr(config_module, "settings", SimpleNamespace(llm_model=""))
    assert current_llm_model() is None
    # 连 llm_model 属性都没有：取属性失败也不允许抛出（落库不能因溯源信息中断）
    monkeypatch.setattr(config_module, "settings", object())
    assert current_llm_model() is None


def test_persist_turn_still_persists_when_model_unconfigured(monkeypatch) -> None:
    """模型名取不到时照常落库（model=None），回答流不受影响。"""
    monkeypatch.setattr(config_module, "settings", SimpleNamespace(llm_model=""))
    store = RecordingChatStore()

    _run_persist(store)

    assert len(store.calls) == 1
    assert store.calls[0]["model"] is None


def test_current_llm_model_reads_env_backed_settings(monkeypatch) -> None:
    """走真实配置装载链路：环境变量注入的模型名能被读取。"""
    # 屏蔽 .env，保证读到的就是本次注入的环境变量（与 conftest 的注入策略无关）
    monkeypatch.setattr(config_module, "load_environment_file", lambda *args, **kwargs: 0)
    monkeypatch.setenv("LLM_MODEL", "model-from-env")
    fresh_settings = config_module.Settings.from_environment()
    monkeypatch.setattr(config_module, "settings", fresh_settings)

    assert current_llm_model() == "model-from-env"
