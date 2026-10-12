# -*- coding: utf-8 -*-
"""单元测试：验证质检 Skill 五大功能与分类路由。直接 `python tests/test_quality_assessor.py` 运行。"""
import os, sys, tempfile, shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quality_assessor import assess, load_config, file_md5
from src.simhash import SimHash


def make_demo_ds(root):
    """构造小型数据集：md/txt/重复文件/相似版本/敏感信息。"""
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "a.md"), "w", encoding="utf-8") as f:
        f.write("# 标题\n" + "这是一段正常的工业文档内容。" * 20)
    with open(os.path.join(root, "b.md"), "w", encoding="utf-8") as f:
        f.write("# 标题\n" + "这是一段正常的工业文档内容。" * 20)  # 与 a.md 完全相同（MD5 重复）
    with open(os.path.join(root, "c.md"), "w", encoding="utf-8") as f:
        f.write("# 标题\n" + "这是一段正常的工业文档内容，略有修改的版本。" * 20)  # 与 a 高度相似（SimHash）
    with open(os.path.join(root, "sens.txt"), "w", encoding="utf-8") as f:
        f.write("联系电话 13800138000，邮箱 test@example.com，身份证号 110101199003078888。")
    with open(os.path.join(root, "notes.txt"), "w", encoding="utf-8") as f:
        f.write("短文档")


def main():
    cfg = load_config()
    with tempfile.TemporaryDirectory() as d:
        make_demo_ds(d)
        report = assess(d, cfg)

    # 断言：五大功能均产出
    assert "format_distribution" in report
    assert "pdf_classification" in report
    assert "length_distribution" in report
    assert report["length_distribution"]["percentiles"]["P50"] >= 0
    assert "duplicates" in report
    assert len(report["duplicates"]["exact_duplicates"]) >= 1, "应检测到 a.md/b.md 精确重复"
    assert report["action_items"]["version_conflicts"], "应检测到 a/c 版本冲突"
    assert report["sensitive_info"]["summary"].get("phone", 0) >= 1
    assert report["sensitive_info"]["summary"].get("email", 0) >= 1
    assert report["sensitive_info"]["summary"].get("id_card", 0) >= 1
    # 列表可操作性：每条含上下文
    for item in report["action_items"]["sensitive_to_review"]:
        assert "context" in item and "file" in item
    # 分类标签
    assert all("label" in v and "route_to" in v for v in report["classification"].values())
    # SimHash 基本正确性
    sh = SimHash()
    assert sh.hamming(sh.hash("你好世界"), sh.hash("你好世界")) == 0
    print("全部单元测试通过 ✓")
    print("  精确重复组数:", len(report["duplicates"]["exact_duplicates"]))
    print("  版本冲突数:", len(report["action_items"]["version_conflicts"]))
    print("  敏感信息命中:", report["sensitive_info"]["summary"])


if __name__ == "__main__":
    main()
