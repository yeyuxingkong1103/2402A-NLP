from backend.app.services.privacy_service import redact_pii


def test_redacts_phone_id_card_and_bank_card():
    source = "我手机号13800138000，身份证110101199003074512，银行卡6222020202020202020。"
    result = redact_pii(source)

    assert "13800138000" not in result.text
    assert "110101199003074512" not in result.text
    assert "6222020202020202020" not in result.text
    assert "[手机号]" in result.text
    assert "[身份证号]" in result.text
    assert "[银行卡号]" in result.text
    assert set(result.redacted_types) >= {"phone", "id_card", "bank_card"}
