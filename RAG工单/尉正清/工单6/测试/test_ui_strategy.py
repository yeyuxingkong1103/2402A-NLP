# 工单编号：人工智能NLP-RAG-混合检索任务
"""检索策略截图：同一个问题分别用三种策略提问，截取界面

用法：python test_ui_strategy.py
截图输出到 测试/截图/策略/
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEV = HERE.parent / "研发"
SHOTS = HERE / "截图" / "策略"
PORT = 7860
PDF1 = r"D:\BW\RAG 工单\附件\招股说明书1.pdf"
PDF2 = r"D:\BW\RAG 工单\附件\招股说明书2.pdf"

# 用一个能体现差异的问题：全文检索单用时召回质量明显更差
QUESTION = "武汉兴图新科电子股份有限公司注册资本是多少？"
STRATEGIES = ["向量检索（召回+重排）", "全文检索", "混合检索"]


def main():
    from playwright.sync_api import sync_playwright

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
                print("[错误] 服务未能启动"); return 1
            page.wait_for_timeout(3000)

            print("==> 上传两份 PDF 并初始化（走缓存）")
            page.locator('input[type="file"]').first.set_input_files([PDF1, PDF2])
            page.wait_for_timeout(2000)
            page.locator('button:has-text("初始化知识库")').first.click()
            for _ in range(90):
                page.wait_for_timeout(2000)
                if "初始化完成" in page.inner_text("body"):
                    break
            SHOTS.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(SHOTS / "00-初始化完成.png"), full_page=True)

            box = page.locator("textarea:enabled").first
            answer_box = page.locator("textarea").nth(4)
            prev = ""
            for label in STRATEGIES:
                print(f"==> 策略：{label}")
                page.locator(f'label:has-text("{label}")').first.click()
                page.wait_for_timeout(500)
                box.fill(QUESTION)
                page.locator('button:has-text("提问")').first.click()
                for _ in range(60):
                    page.wait_for_timeout(1000)
                    cur = answer_box.input_value()
                    if cur and cur != prev:
                        break
                page.wait_for_timeout(1200)
                prev = answer_box.input_value()
                page.screenshot(path=str(SHOTS / f"{label[:4]}.png"), full_page=True)
                print(f"  已截图 {label}")

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("==> 完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
