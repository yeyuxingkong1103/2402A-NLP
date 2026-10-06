from app.services.prompt_builder import build_prompt


def test_build_prompt_contains_all_sections():
    character = {
        "name": "小白", "persona": "温柔的图书馆管理员",
        "worldview": "近未来", "relationship": "老朋友",
        "hidden_setting": "其实是猫", "sample_dialogue": "示例",
    }
    msgs = build_prompt(character, ["用户喜欢猫"], "早前摘要", [{"role": "user", "content": "你好"}])
    system = msgs[0]
    assert system["role"] == "system"
    assert "小白" in system["content"]
    assert "隐藏设定" in system["content"] and "禁止主动透露" in system["content"]
    assert "用户喜欢猫" in system["content"]
    assert msgs[-1] == {"role": "user", "content": "你好"}


def test_build_prompt_omits_optional_sections():
    character = {"name": "A", "persona": "", "worldview": "", "relationship": "", "hidden_setting": "", "sample_dialogue": ""}
    msgs = build_prompt(character, [], None, [])
    # 空字段不产出多余标题
    assert "## 隐藏设定" not in msgs[0]["content"]
