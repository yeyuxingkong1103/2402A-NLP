# 工单编号：人工智能NLP-RAG-Query 理解优化任务
"""多轮对话截图：在界面上按工单演示的 5 轮对话依次提问，截取对话历史

用法：python test_ui_dialogue.py
截图输出到 测试/截图/对话/
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dialogue_truth import DIALOGUE                                  # noqa: E402

ROOT = HERE.parent
DEV = ROOT / "研发"
SHOTS = HERE / "截图" / "对话"
PORT = 7860
PDF1 = r"D:\BW\RAG 工单\附件\招股说明书1.pdf"
PDF2 = r"D:\BW\RAG 工单\附件\招股说明书2.pdf"


def main():
    from playwright.sync_api import sync_playwright

    print("==> 启动服务 ...")
    proc = subprocess.Popen([sys.executable, "-u", "app.py"], cwd=str(DEV),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 1600})
            for _ in range(40):
                try:
                    page.goto(f"http://127.0.0.1:{PORT}/", timeout=3000)
                    break
                except Exception:                                  # noqa: BLE001
                    time.sleep(2)
            else:
                print("[错误] 服务未能启动"); return 1
            page.wait_for_timeout(3000)

            print("==> 上传两份 PDF 并初始化知识库（走缓存）")
            page.locator('input[type="file"]').first.set_input_files([PDF1, PDF2])
            page.wait_for_timeout(2000)
            page.locator('button:has-text("初始化知识库")').first.click()
            for _ in range(90):
                page.wait_for_timeout(2000)
                if "初始化完成" in page.inner_text("body"):
                    break
            page.screenshot(path=str(SHOTS / "00-初始化完成.png"), full_page=True)

            box = page.locator("textarea:enabled").first
            answer_box = page.locator("textarea").nth(4)
            SHOTS.mkdir(parents=True, exist_ok=True)

            print(f"==> 依次提出 {len(DIALOGUE)} 轮问题")
            prev = ""
            for item in DIALOGUE:
                box.fill(item["question"])
                page.locator('button:has-text("提问")').first.click()
                for _ in range(40):                                # 等新答案出现
                    page.wait_for_timeout(1000)
                    cur = answer_box.input_value()
                    if cur and cur != prev:
                        break
                page.wait_for_timeout(1200)
                prev = answer_box.input_value()

                # 整页截图：对话历史 + 回答 + 元信息栏（含改写过程）一并留证。
                # 不用元素定位，Gradio 各版本的 testid 不一致，容易踩空。
                page.screenshot(path=str(SHOTS / f"第{item['turn']}轮.png"), full_page=True)
                print(f"  第 {item['turn']} 轮已截图", flush=True)

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
