# 测试 check_style.py：它要在验收时作为"代码规范零告警"的证据，所以它自己必须可靠
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from check_style import count_code_lines, find_uncommented_lines, check_file


def test_count_code_lines_ignores_blank():
    # 空行不计入非空行统计——这是硬约束的判定口径
    assert count_code_lines("a = 1\n\n\nb = 2\n") == 2


def test_find_uncommented_lines_flags_bare_code():
    # 一行代码若其上方与行尾都没有注释，应被判定为"无注释行"
    src = "def f():\n    return 1\n"
    assert find_uncommented_lines(src) == [1, 2]


def test_find_uncommented_lines_passes_when_preceded_by_comment():
    src = "# 说明为什么这么做\ndef f():\n    # 说明为什么返回 1\n    return 1\n"
    assert find_uncommented_lines(src) == []


def test_check_file_reports_oversize(tmp_path):
    # 301 行非空代码必须被标记超限
    p = tmp_path / "big.py"
    p.write_text("\n".join(f"x{i} = {i}  # 注释" for i in range(301)), encoding="utf-8")
    result = check_file(p)
    assert result["oversize"] is True
    assert result["code_lines"] == 301
