# 工单编号：人工智能NLP-RAG-功能测试及评估
"""界面测试截图：在界面上跑几道代表性题目，把结果截下来

用法：python test_ui_ccf.py
截图输出到 测试/截图/

选这三道题是有讲究的，分别对应测试报告第五节里的三种情况：
  Q2  分析型，检索 100% 命中        —— 正常表现长什么样
  Q4  事实型，三个指标只召回两个    —— 部分缺失时系统的行为（如实说答不了）
  Q10 跨文档，三家银行一个都没召回  —— 检索失败长什么样
另外单独截一张「检索到的上下文」，让检索结果本身可见，
而不只是看最终答案。

界面上的 textarea 顺序是固定的（0 状态、1 知识库消息、2 提问框、3 回答、
4 上下文、5 元信息、6 反馈），Gradio 不给这些控件稳定的 id，
只能按下标取；下面的 _TEXTAREA 常量就是这套下标。
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEV = HERE.parent / "研发"
SHOTS = HERE / "截图"
PORT = 7860

Q_QUESTION, Q_ANSWER, Q_CONTEXT, Q_META = 2, 3, 4, 5
CASES = [
    (2, "02-Q2-分析型-检索命中"),
    (4, "03-Q4-事实型-部分指标未召回"),
    (10, "04-Q10-跨文档-银行完全未召回"),
]


def wait_text(page, idx, prev="", timeout=90):
    """等某个 textarea 的值发生变化（Gradio 是整块替换，不变化就是还没回）。"""
    for _ in range(timeout):
        page.wait_for_timeout(1000)
        cur = page.locator("textarea").nth(idx).input_value()
        if cur and cur != prev:
            return cur
    return page.locator("textarea").nth(idx).input_value()


def open_dropdown(page, index, tries=3):
    """展开 Gradio 下拉框。

    实测踩过的坑：下拉框被 Esc 关掉之后，再点输入框**打不开**
    （点一次没反应，li 数一直是 0），要多点几次才恢复。
    所以这里重试而不是点一次就算。
    """
    for _ in range(tries):
        page.locator('input[role="listbox"]').nth(index).click()
        page.wait_for_timeout(800)
        if page.locator("li").count():
            return True
    return False


def pick_dropdown(page, index, text):
    """在下拉框里选一项。"""
    if not open_dropdown(page, index):
        raise RuntimeError(f"下拉框 {index} 打不开")
    page.locator(f'li:has-text("{text}")').first.click()
    page.wait_for_timeout(700)


def main():
    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-u", "app.py"], cwd=str(DEV),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 1700})
            for _ in range(40):
                try:
                    page.goto(f"http://127.0.0.1:{PORT}/", timeout=3000)
                    break
                except Exception:                                  # noqa: BLE001
                    time.sleep(2)
            else:
                print("[错误] 服务未能启动")
                return 1
            page.wait_for_timeout(6000)

            print("==> 加载 ccf_competition 知识库")
            pick_dropdown(page, 0, "ccf_competition")
            page.locator('button:has-text("加载 / Load")').first.click()
            for _ in range(30):
                page.wait_for_timeout(1000)
                if "已加载知识库" in page.locator("textarea").nth(1).input_value():
                    break
            print("   ", page.locator("textarea").nth(1).input_value())
            page.screenshot(path=str(SHOTS / "00-知识库已加载.png"), full_page=True)

            # 展开预置问题下拉框把 10 道题列出来。截完**不要按 Esc 关它** ——
            # 关掉之后这个下拉框要连点好几次才打得开，后面选题目会卡住。
            print("==> 截图：预置的 10 道测试题")
            open_dropdown(page, 1)
            page.screenshot(path=str(SHOTS / "01-预置的10道测试题.png"), full_page=True)

            prev = ""
            for qid, name in CASES:
                print(f"==> Q{qid} {name}")
                if qid != CASES[0][0]:          # 第一题的列表还开着，直接用
                    pick_dropdown(page, 1, f"{qid} |")
                else:
                    page.locator(f'li:has-text("{qid} |")').first.click()
                    page.wait_for_timeout(700)
                page.locator('button:has-text("提问 / Ask")').first.click()
                answer = wait_text(page, Q_ANSWER, prev)
                prev = answer
                page.wait_for_timeout(1500)
                page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True)
                print("   答案开头：", answer[:60].replace("\n", " "))
                print("   元信息：", page.locator("textarea").nth(Q_META).input_value()[:160])

            # 单独截一张检索上下文（Q10 那轮上下文还在，正好展示「什么都没召回」）
            ctx = page.locator("textarea").nth(Q_CONTEXT).input_value()
            print("==> 检索上下文截图，长度", len(ctx))
            page.locator("textarea").nth(Q_CONTEXT).scroll_into_view_if_needed()
            page.wait_for_timeout(800)
            page.screenshot(path=str(SHOTS / "05-Q10-检索到的上下文.png"), full_page=True)

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("==> 完成，截图在", SHOTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
