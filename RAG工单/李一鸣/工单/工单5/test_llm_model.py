from llm_model import ModelFactory


def test_factory_returns_requested_model():
    model = ModelFactory.get_model("deepseek-chat", temperature=0.0)
    assert model.config.model_type == "deepseek-chat"
    assert model.config.temperature == 0.0
