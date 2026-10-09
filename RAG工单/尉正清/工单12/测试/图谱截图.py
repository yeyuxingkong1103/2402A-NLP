# 工单编号：人工智能NLP-RAG项目-LightRAG优化
"""给 neo4j 里的知识图谱截图（产出物 2 的可视化证据）

工单的格式要求写明「测试：图片 一定要多截图」。本工单的核心产出物是知识图谱，
它的直接证据就是 neo4j Browser 里的画面 —— 所以这个脚本驱动浏览器登录
neo4j、依次执行几条 Cypher 并截图，而不是自己用绘图库画一张（画的图证明不了
图谱真的建出来了）。

用法（需要 Docker 里的 neo4j 已启动，且 rag_gd 环境有 playwright）：
    D:/Anaconda/envs/rag_gd/python.exe 图谱截图.py
截图输出到 测试/截图/
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
SHOTS = HERE / "截图"
URL = "http://localhost:7474/browser/"
USER, PWD = "neo4j", "neo4j123"

# (文件名, Cypher, 说明)。顺序即编号顺序 —— 从「整体规模」到「具体检索」，
# 看的人顺着往下就能建立印象。
# 节点数刻意压到 25 以内：neo4j Browser 会按节点数自动缩放，
# 给 50+ 个节点时标签会小到看不清，截图就失去意义了。
SHOTS_LIST = [
    ("00-图谱总览-实体与关系",
     "MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 25",
     "一小片连通的子图。左侧栏同时显示全库规模：5,300 实体 / 8,301 关系"),

    ("01-实体类型分布",
     "MATCH (n) WHERE n.entity_type IS NOT NULL "
     "RETURN n.entity_type AS 实体类型, count(*) AS 数量 "
     "ORDER BY 数量 DESC",
     "验收点 1 的直接证据：12 类招股书专用类型各自的数量"),

    ("02-兴图新科-发行人子图",
     "MATCH (a {entity_id:'武汉兴图新科电子股份有限公司'})-[r]-(b) "
     "RETURN a, r, b LIMIT 20",
     "以发行人为中心的一跳邻域：子公司、董监高、资质、客户"),

    ("03-控股关系检索",
     "MATCH (a)-[r]->(b) WHERE r.keywords CONTAINS 'Controlling Shareholder' "
     "RETURN a, r, b LIMIT 15",
     "按关系语义检索。关系类型存在边的 keywords 属性里，"
     "neo4j 里所有边的基础类型都是 DIRECTED（LightRAG 的存储约定）"),

    ("04-力源信息-发行人子图",
     "MATCH (a {entity_id:'武汉力源信息技术股份有限公司'})-[r]-(b) "
     "RETURN a, r, b LIMIT 20",
     "第二份招股书的发行人。对应第 1~4 题的关联方问题"),

    ("05-程家明-人物子图",
     "MATCH (a {entity_id:'程家明'})-[r]-(b) RETURN a, r, b LIMIT 18",
     "对应第 531 题「法定代表人是谁」—— 图谱直接给出人物及其全部关联"),

    ("06-图谱规模统计",
     "MATCH (n) WHERE n.entity_id IS NOT NULL WITH count(n) AS 实体数 "
     "MATCH ()-[r]->() RETURN 实体数, count(r) AS 关系数",
     "总量核对"),

    ("07-带数值的实体描述",
     "MATCH (n) WHERE n.entity_id CONTAINS '5,520' "
     "RETURN n.entity_id AS 实体, n.entity_type AS 类型, "
     "n.description AS 描述 LIMIT 8",
     "第 543 题「注册资本」：数值被写进了实体描述（自定义 schema 的第 4 条规则）"),

    ("08-军用领域相关实体",
     "MATCH (n) WHERE n.entity_id CONTAINS '军用' "
     "RETURN n.entity_id AS 实体, n.entity_type AS 类型, "
     "n.description AS 描述 LIMIT 10",
     "对应第 33、260、957 题的军用领域系列。以表格看描述更清楚"),
]


def login(page):
    """neo4j Browser 停在连接页时，填账号密码并提交。已连上就直接返回。"""
    if page.locator("input[name='username']").count() == 0:
        return
    page.fill("input[name='username']", USER)
    page.fill("input[name='password']", PWD)
    page.click("button[type='submit']")
    page.wait_for_timeout(5000)


def run_cypher(page, cypher):
    """在编辑器里输入 Cypher 并执行（Ctrl+Enter 是 neo4j 的执行快捷键）。"""
    editor = page.locator(".cm-content").first
    editor.click()
    page.keyboard.press("Control+A")
    page.keyboard.press("Delete")
    # 换行会被 CodeMirror 当成执行，所以压成一行
    page.keyboard.type(" ".join(cypher.split()), delay=1)
    page.wait_for_timeout(400)
    page.keyboard.press("Control+Enter")
    page.wait_for_timeout(4500)          # 等图渲染完


def main():
    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1680, "height": 1050})
        page.goto(URL, wait_until="networkidle", timeout=90000)
        page.wait_for_timeout(3000)
        login(page)

        for name, cypher, note in SHOTS_LIST:
            print(f"[截图] {name} —— {note}", flush=True)
            try:
                run_cypher(page, cypher)
                page.screenshot(path=str(SHOTS / f"{name}.png"))
            except Exception as exc:                       # noqa: BLE001
                print(f"    ✗ 失败：{type(exc).__name__}: {str(exc)[:120]}")
        browser.close()

    got = sorted(SHOTS.glob("*.png"))
    print(f"\n完成：{len(got)} 张，输出到 {SHOTS}")
    for g in got:
        print(f"  {g.name}  ({g.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
