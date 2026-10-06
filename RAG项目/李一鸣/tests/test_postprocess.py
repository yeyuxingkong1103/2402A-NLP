from app.rag.postprocess import postprocess_answer


def test_postprocess_removes_thinking_and_fences():
    answer = "<think>internal</think>\n答案：```markdown\n结论\n```"
    assert postprocess_answer(answer) == "结论"
