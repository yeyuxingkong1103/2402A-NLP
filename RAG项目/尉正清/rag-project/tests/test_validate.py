# tests/test_validate.py
"""生成结果校验的单元测试。

只测纯逻辑（引用抽取与归一化），不依赖服务与大模型。
"""

from app.core.validate_service import ValidateService, extract_citations


class TestExtraction:
    """引用抽取：各种写法都要能认出来。"""

    def test_quoted_full_name(self):
        cites = extract_citations("根据《中华人民共和国刑法》第一百三十三条规定……")
        assert len(cites) == 1
        assert cites[0]["law"] == "中华人民共和国刑法"
        assert cites[0]["article"] == "第一百三十三条"

    def test_bare_arabic_number(self):
        """「刑法第133条」这种口语写法要归一化成中文条号。"""
        cites = extract_citations("按刑法第133条，交通肇事逃逸……")
        assert cites and cites[0]["article"] == "第一百三十三条"

    def test_civil_code_short_name(self):
        """《民法典》不以「法」结尾，抽取逻辑不能漏。"""
        cites = extract_citations("依据民法典第1079条，法院应先行调解。")
        assert len(cites) == 1
        assert cites[0]["law"] == "中华人民共和国民法典"
        assert cites[0]["article"] == "第一千零七十九条"

    def test_sub_article_preserved(self):
        """「之一」这类增补条款不能被截断。"""
        cites = extract_citations("《中华人民共和国刑法》第一百三十三条之一规定……")
        assert cites[0]["article"] == "第一百三十三条之一"

    def test_longer_alias_wins(self):
        """「劳动合同法」不能被「劳动法」抢先匹配。"""
        cites = extract_citations("《中华人民共和国劳动合同法》第八十二条……")
        assert cites[0]["law"] == "中华人民共和国劳动合同法"

    def test_dedup_same_citation(self):
        """同一法条重复提及只算一次。"""
        text = "《中华人民共和国刑法》第七十四条……如前所述，第七十四条明确……"
        cites = extract_citations(text)
        assert len(cites) == 1

    def test_no_citation(self):
        assert extract_citations("今天天气不错。") == []

    def test_empty_input(self):
        assert extract_citations("") == []
        assert extract_citations(None) == []


class TestNote:
    """提示文案。"""

    def test_no_note_when_all_verified(self):
        assert ValidateService.build_note({"unverified": []}) == ""

    def test_note_lists_unverified(self):
        note = ValidateService.build_note({
            "unverified": [{"raw": "《中华人民共和国刑法》第九百九十九条"}]})
        assert "第九百九十九条" in note
        assert "未能" in note or "核对" in note

    def test_note_truncates_long_list(self):
        note = ValidateService.build_note({"unverified": [
            {"raw": "《A法》第一条"}, {"raw": "《B法》第二条"},
            {"raw": "《C法》第三条"}, {"raw": "《D法》第四条"}]})
        assert "等 4 处" in note
