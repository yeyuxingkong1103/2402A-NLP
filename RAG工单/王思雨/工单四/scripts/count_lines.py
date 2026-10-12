# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 文件说明：统计有效代码行数（非空行、且非纯注释行），用于如实申报工单的 300 行约束。
# 用法：python scripts/count_lines.py
import ast                                                                   # 用 AST 精确判断"哪些行是真实代码行"
import sys                                                                   # 退出
from pathlib import Path                                                     # 路径

ROOT = Path(__file__).resolve().parent.parent                                # 项目根


def effective(path: Path) -> tuple:                                          # 单文件有效行
    """以 Python 语法树节点覆盖的行号为准：字符串文档串、注释、空行都不计入。

    之所以不用"以 # 开头就跳过"这种朴素规则：多行字符串（提示词模板）里的行既不是注释也不是空行，
    朴素规则会把它们当成代码，把行数虚报；用 AST 才数得准。
    """
    src = path.read_text(encoding="utf-8")                                   # 源码
    lines = src.splitlines()                                                 # 全部行
    code_rows = set()                                                        # 真实代码行号集合
    try: tree = ast.parse(src)                                               # 解析
    except SyntaxError: return len(lines), 0, 0                              # 解析不了就如实报全量
    for node in ast.walk(tree):                                              # 遍历全部节点
        # 只取"语句"与"表达式"类节点：模块/函数/类的定义行由它们的子语句覆盖，避免重复计
        if isinstance(node, (ast.stmt, ast.expr, ast.excepthandler, ast.arguments, ast.keyword)):  # 有行号的语法单元
            if hasattr(node, "lineno"):                                      # 有起始行
                code_rows.add(node.lineno)                                   # 记下该行
    blank = sum(1 for l in lines if not l.strip())                           # 空行
    comment = sum(1 for l in lines if l.strip().startswith("#"))              # 纯注释行
    return len(code_rows), blank, comment                                    # 有效行、空行、注释行


def main() -> None:                                                          # 主流程
    files = sorted([p for p in ROOT.rglob("*.py") if "_frames" not in str(p)])  # 全部 Python 文件
    total_e = total_a = 0                                                    # 汇总
    print(f"{'文件':<34}{'有效':>6}{'空行':>6}{'注释':>7}{'总行':>7}")           # 表头
    print("-" * 62)                                                         # 分隔
    for f in files:                                                          # 逐文件
        e, b, c = effective(f)                                               # 统计
        raw = len(f.read_text(encoding="utf-8").splitlines())                 # 总行
        total_e += e; total_a += raw                                          # 累计
        print(f"{str(f.relative_to(ROOT)):<34}{e:>6}{b:>6}{c:>7}{raw:>7}")    # 输出
    print("-" * 62)                                                         # 分隔
    print(f"{'合计':<34}{total_e:>6}{'':>6}{'':>7}{total_a:>7}")               # 合计
    core = sum(effective(ROOT / p)[0] for p in ["config.py", "backend/main.py"])  # 核心逻辑：配置 + 接口与流水线
    print(f"\n工单要求的 300 行约束：")                                          # 口径说明
    print(f"  全部 Python 有效代码行      = {total_e}")                        # 全量
    print(f"  核心逻辑（config+main）有效行 = {core}")                           # 核心
    print(f"  判定：{'全部达标' if total_e <= 300 else f'整体超出 {total_e - 300} 行，核心逻辑{core}行'}")


if __name__ == "__main__":                                                    # 入口
    main()                                                                    # 执行
