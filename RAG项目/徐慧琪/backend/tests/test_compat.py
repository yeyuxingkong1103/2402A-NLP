# compat 的规避效果只能起**干净子进程**来测：本进程里 pandas 早被 pymilvus
# 顺带拉进了 sys.modules（app.db.milvus 的顶层导入链），在本进程里无论怎么
# 摆弄导入顺序都测不出规避有没有生效——子进程里刻意不碰 app.db.milvus。
import ast
import pathlib
import subprocess
import sys

BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]

# 子进程按真实调用点的方式走一遍：先 ensure_import_order 再 import
# FlagEmbedding。开头那条 assert 钉住测试前提——pandas 还没进 sys.modules，
# 规避确实处在被测状态；前提没了（比如有 sitecustomize 提前导了 pandas），
# 本测会退化成恒绿而不自知，宁可当场红（项目对"不可能失败的测试"已有八次
# 前车之鉴）。不在这里断言"pandas 已被拉进来"：那样它会在段错误之前先失败，
# 红因从"没规避"变成"没导 pandas"，反而盖掉真正的失败现场
CHILD_CODE = """
import sys

assert "pandas" not in sys.modules, "子进程不干净：pandas 已被先行导入，本测失去判别力"
from app.compat import ensure_import_order

ensure_import_order()
import FlagEmbedding
"""


def test_ensure_import_order_keeps_flagembedding_import_alive():
    """裸 import FlagEmbedding 会段错误；先走规避则子进程正常退出。

    判别力已用忠实变异验证：把 ensure_import_order 的函数体改成 pass，
    本测变红（子进程段错误）——证明它守的是规避本身，不是"子进程怎么都能活"。
    """
    proc = subprocess.run([sys.executable, "-c", CHILD_CODE],
                          cwd=BACKEND_DIR, capture_output=True, timeout=300)
    # 输出按字节收回再容错解码：子进程被访问违例杀掉时 stderr 可能为空，
    # 正常报错时也可能带非 GBK 字符，用 text=True 会连解码异常一起抛出来
    stderr = (proc.stderr or b"").decode("utf-8", errors="replace")[-2000:]
    assert proc.returncode == 0, (
        f"子进程退出码 {proc.returncode}（段错误/访问违例 0xC0000005，"
        f"bash 里显示 139，说明规避未生效）：\n{stderr}")


# 上面那条子进程测试守的是"机制"（ensure_import_order 本身有效），从不经过两个
# 真实调用点——而全量跑时 test_embed 的收集期就由 pymilvus 把 pandas 装进了
# sys.modules，删掉任一调用点全量仍绿、没有任何测试变红。这里补"调用点"防线：
# 直接对源码做 AST 顺序断言。不加载模型、不受收集顺序影响，确定性强。
CALL_SITES = [
    # (源码路径, 函数名, 该函数体内的重导入模块名)
    (BACKEND_DIR / "app" / "retrieval" / "rerank.py", "load_reranker", "sentence_transformers"),
    (BACKEND_DIR / "app" / "ingest" / "embed.py", "load_model", "FlagEmbedding"),
]


def _guard_before_import_linenos(path: pathlib.Path, func_name: str,
                                 module_name: str) -> tuple[int, int]:
    """返回 (规避调用行号, 重导入行号)；两个锚点缺一即红，不许静默退化。

    用 ast 而不是文本查找：这两个名字在注释里也会出现（调用点上方就写着
    "必须先于 ... 调用"），字符串匹配会把注释误当代码。只认真正的语法节点。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    funcs = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name == func_name]
    assert funcs, f"{path} 顶层找不到 {func_name}——断言失去目标，先修本测"
    calls: list[int] = []
    imports: list[int] = []
    for node in ast.walk(funcs[0]):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "ensure_import_order"):
            calls.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module == module_name:
            imports.append(node.lineno)
    assert calls, f"{func_name} 里找不到 ensure_import_order() 调用"
    assert imports, f"{func_name} 里找不到 from {module_name} import ..."
    return min(calls), min(imports)


def test_call_sites_guard_heavy_imports_before_they_run():
    """真实调用点的先后顺序：规避调用必须早于函数体内的重导入。

    判别力已用忠实变异验证：删掉调用点、或把它挪到重导入之后，两文件共四种
    坏法本测全红，而子进程测试对它们全绿——这正是本测存在的理由：将来只删
    一处调用点时，只有它能拦（全量跑因收集期已带 pandas 不会崩）。
    """
    for path, func_name, module_name in CALL_SITES:
        call_line, import_line = _guard_before_import_linenos(path, func_name,
                                                              module_name)
        assert call_line < import_line, (
            f"{path.name} 的 {func_name}：ensure_import_order() 在第 {call_line} 行，"
            f"不早于 from {module_name} 的第 {import_line} 行——重导入会先执行，规避落空")
