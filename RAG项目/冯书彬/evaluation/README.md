# Evaluation

## 婚姻家庭 MVP 评测集

Task 14 新增 `evaluation/datasets/marriage_family_mvp_100.jsonl`，用于验证法律客服助手在婚姻家庭 MVP 范围内的回答边界、引用依据和风险处理。

### 数据分布

- `离婚与婚姻关系`：40 条
- `抚养与探望`：30 条
- `夫妻财产与债务`：30 条

### 字段说明

每条 JSONL 记录包含：

- `case_id`：评测用例编号。
- `domain`：三类婚姻家庭子域之一。
- `user_question`：虚构用户问题。
- `fictional_facts`：匿名虚构事实，不复制真实当事人信息。
- `facts_to_confirm`：回答前应追问或核对的关键事实。
- `risk_level`：`normal`、`medium` 或 `high`。
- `expected_legal_basis`：期望引用的法律依据或规则来源。
- `expected_answer_points`：期望回答覆盖的要点。
- `must_avoid`：回答中必须避免的内容。
- `scoring_rubric`：评测维度权重。
- `source_refs`：只读来源文件路径。

### 校验命令

```bash
python evaluation/marriage_family_mvp/validate_dataset.py evaluation/datasets/marriage_family_mvp_100.jsonl
```

期望输出：

```text
PASS: 100 records validated with counts 40/30/30
```

### 单元测试

```bash
python -m pytest evaluation/marriage_family_mvp/test_validate_dataset.py -q
```

### 注意事项

- `C:\Users\bin\Desktop\public` 只作为只读资料来源，不应被评测脚本修改。
- 数据集事实均为匿名虚构场景，不应复制真实案例中的姓名、身份证、住址、案号等个人信息。
- 当前 `evaluate_mvp.py` 是确定性评估骨架，后续可接入人工复核或模型辅助评测。
