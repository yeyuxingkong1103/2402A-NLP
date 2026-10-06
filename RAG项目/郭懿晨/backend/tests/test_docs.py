def test_v2_docs_exist():
    from pathlib import Path

    expected_files = [
        Path("docs/需求说明.md"),
        Path("docs/版本迭代.md"),
        Path("docs/架构/架构图-v2.md"),
        Path("docs/v2/spec.md"),
        Path("docs/v2/plan.md"),
        Path("docs/v2/tasks.md"),
        Path("docs/v2/checklist.md"),
        Path("data/README.md"),
        Path("eval/sets/README.md"),
        Path("eval/baseline/README.md"),
    ]

    for path in expected_files:
        assert path.exists()
