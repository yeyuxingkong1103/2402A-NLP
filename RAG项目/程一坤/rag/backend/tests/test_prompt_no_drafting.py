"""批次 12-4：系统提示词必须包含"不代写法律文书"边界。"""

from app.chat.prompt_builder import SYSTEM_PROMPT


def test_system_prompt_forbids_document_drafting() -> None:
    """refusal-080 暴露的缺口：模型对"写仲裁申请书"直接代写了。

    提示词铁律须明确：不代写文书，改为说明不做代写 + 提供条文要点。
    """
    assert "不代写法律文书" in SYSTEM_PROMPT
    assert "申请书" in SYSTEM_PROMPT
    assert "法律信息问答" in SYSTEM_PROMPT
