from evaluation.marriage_family_mvp.validate_dataset import validate_dataset, validate_record


def _valid_record(case_id="MF-DIV-001", domain="离婚与婚姻关系"):
    return {
        "case_id": case_id,
        "domain": domain,
        "user_question": "我想离婚，需要先准备什么？",
        "fictional_facts": "用户甲与配偶乙已登记结婚，暂未说明所在地和是否有子女。",
        "facts_to_confirm": ["所在省市", "是否有未成年子女", "是否协商一致"],
        "risk_level": "normal",
        "expected_legal_basis": ["中华人民共和国民法典 第一千零七十六条"],
        "expected_answer_points": ["说明登记离婚和诉讼离婚路径", "提示补充关键信息"],
        "must_avoid": ["保证一定能离婚", "生成离婚协议书正文"],
        "scoring_rubric": {"basis_accuracy": 2, "boundary": 2, "follow_up": 1},
        "source_refs": ["civil_code_articles/civil_code_articles.txt"],
    }


def test_valid_record_passes():
    assert validate_record(_valid_record()) == []


def test_missing_required_field_fails():
    record = _valid_record()
    record.pop("source_refs")

    assert validate_record(record) == ["缺少字段：source_refs"]


def test_dataset_requires_exact_domain_counts():
    records = []
    records.extend(_valid_record(f"MF-DIV-{index:03d}", "离婚与婚姻关系") for index in range(1, 41))
    records.extend(_valid_record(f"MF-CUS-{index:03d}", "抚养与探望") for index in range(1, 31))
    records.extend(_valid_record(f"MF-PROP-{index:03d}", "夫妻财产与债务") for index in range(1, 31))

    assert validate_dataset(records) == []


def test_dataset_reports_wrong_count_and_duplicates():
    records = [_valid_record("MF-DIV-001", "离婚与婚姻关系"), _valid_record("MF-DIV-001", "离婚与婚姻关系")]

    errors = validate_dataset(records)

    assert "数据集必须正好 100 条，当前为 2 条" in errors
    assert "case_id 重复：MF-DIV-001" in errors
