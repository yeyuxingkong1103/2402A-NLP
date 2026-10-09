# 工单编号：人工智能NLP-RAG-金融问答系统部署
"""给「部署后的服务」截图

和工单7 的 test_ui_ccf.py 的区别：那个会自己拉起本地 app.py，
这个**只连接已经在跑的容器服务**——工单10 要证明的是容器里那份服务能用，
连本地进程截的图不算数。

用法：
    先在容器里起服务，再执行
    python 部署截图.py                    # 默认连 http://127.0.0.1:7860
    python 部署截图.py --url http://x:7860
截图输出到 测试/截图/
"""
import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
SHOTS = HERE / "截图"

# 界面上的 textarea 顺序（Gradio 不给稳定 id，只能按下标取）
Q_QUESTION, Q_ANSWER, Q_CONTEXT, Q_META = 2, 3, 4, 5

# 三道题分别覆盖：事实型命中 / 跨文档对比 / 界面预置题
CASES = [
    (3, "02-事实型-平安银行拨备覆盖率"),
    (9, "03-跨文档-两家券商营业收入"),
    (10, "04-跨文档-银行与保险"),
]


def open_dropdown(page, index, tries=3):
    """展开下拉框。被 Esc 关掉后一次点不开，要多点几次。"""
    for _ in range(tries):
        page.locator('input[role="listbox"]').nth(index).click()
        page.wait_for_timeout(800)
        if page.locator("li").count():
            return True
    return False


def wait_answer(page, prev="", timeout=120):
    for _ in range(timeout):
        page.wait_for_timeout(1000)
        cur = page.locator("textarea").nth(Q_ANSWER).input_value()
        if cur and cur != prev:
            return cur
    return page.locator("textarea").nth(Q_ANSWER).input_value()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1750})

        print(f"==> 连接部署后的服务 {args.url}")
        for i in range(30):
            try:
                page.goto(f"{args.url}/", timeout=5000)
                break
            except Exception:                                       # noqa: BLE001
                time.sleep(3)
        else:
            print("[错误] 服务连不上，确认容器在跑且端口已映射")
            return 1
        page.wait_for_timeout(5000)
        page.screenshot(path=str(SHOTS / "00-服务首页.png"), full_page=True)

        # 加载知识库
        print("==> 加载知识库")
        open_dropdown(page, 0)
        page.locator('li:has-text("ccf_competition")').first.click()
        page.wait_for_timeout(600)
        page.locator('button:has-text("加载 / Load")').first.click()
        for _ in range(30):
            page.wait_for_timeout(1000)
            if "已加载知识库" in page.locator("textarea").nth(1).input_value():
                break
        print("   ", page.locator("textarea").nth(1).input_value())

        print("==> 展开预置问题")
        open_dropdown(page, 1)
        page.screenshot(path=str(SHOTS / "01-预置问题列表.png"), full_page=True)

        prev = ""
        for idx, (qid, name) in enumerate(CASES):
            print(f"==> Q{qid} {name}")
            if idx == 0:
                page.locator(f'li:has-text("{qid} |")').first.click()  # 列表还开着
            else:
                open_dropdown(page, 1)
                page.locator(f'li:has-text("{qid} |")').first.click()
            page.wait_for_timeout(700)
            page.locator('button:has-text("提问 / Ask")').first.click()
            answer = wait_answer(page, prev)
            prev = answer
            page.wait_for_timeout(1500)
            page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True)
            print("   答案：", answer[:70].replace("\n", " "))
            print("   元信息：", page.locator("textarea").nth(Q_META).input_value()[:150])

        browser.close()
    print("==> 完成，截图在", SHOTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
