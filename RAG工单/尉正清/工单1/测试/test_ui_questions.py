# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""逐题验收截图：在界面上依次提问工单指定的 10 个问题，每题截图一张

用法：python test_ui_questions.py [PDF路径]
截图输出到 测试/截图/逐题/
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEV = ROOT / "研发"
sys.path.insert(0, str(DEV))

from config import TEST_QUESTIONS                       # noqa: E402

SHOTS = Path(__file__).resolve().parent / "截图" / "逐题"
PORT = 7860
PDF = sys.argv[1] if len(sys.argv) > 1 else r"D:\BW\RAG 工单\附件\招股说明书1.pdf"


def clip_of(page, answer_box, meta_box):
    """取「回答 + 元信息」两块区域的并集，截出来更紧凑好读。"""
    a, m = answer_box.bounding_box(), meta_box.bounding_box()
    if not a or not m:
        return None
    left = min(a["x"], m["x"]) - 8
    top = a["y"] - 8
    right = max(a["x"] + a["width"], m["x"] + m["width"]) + 8
    bottom = m["y"] + m["height"] + 8
    return {"x": max(left, 0), "y": max(top, 0),
            "width": right - left, "height": bottom - top}


def main():
    from playwright.sync_api import sync_playwright

    print("==> 启动服务 ...")
    proc = subprocess.Popen([sys.executable, "-u", "app.py"], cwd=str(DEV),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 1200})

            for _ in range(40):
                try:
                    page.goto(f"http://127.0.0.1:{PORT}/", timeout=3000)
                    break
                except Exception:                              # noqa: BLE001
                    time.sleep(2)
            else:
                print("[错误] 服务未能启动"); return 1
            page.wait_for_timeout(2500)

            print("==> 上传 PDF 并初始化知识库（走缓存）")
            page.locator('input[type="file"]').first.set_input_files(PDF)
            page.wait_for_timeout(1500)
            page.locator('button:has-text("初始化知识库")').first.click()
            for _ in range(60):
                page.wait_for_timeout(2000)
                if "完成" in page.inner_text("body"):
                    break

            question_box = page.locator("textarea:enabled").first
            answer_box = page.locator("textarea").nth(3)   # 回答
            meta_box = page.locator("textarea").nth(5)     # 元信息
            SHOTS.mkdir(parents=True, exist_ok=True)

            print(f"==> 依次提问 {len(TEST_QUESTIONS)} 个问题")
            prev = ""
            for item in TEST_QUESTIONS:
                question_box.fill(item["question"])
                page.locator('button:has-text("提问")').first.click()

                # 关键：新答案生成期间，答案框里还留着【上一题】的文本。
                # 必须先等它变成「与上一题不同」的内容，否则会截到上一题的答案。
                for _ in range(40):
                    page.wait_for_timeout(1000)
                    cur = answer_box.input_value()
                    if cur and cur != prev:
                        break
                page.wait_for_timeout(1200)          # 再等一拍，确保文本已完整写入
                prev = answer_box.input_value()

                path = SHOTS / f"{item['id']}.png"
                clip = clip_of(page, answer_box, meta_box)
                data = page.screenshot(clip=clip) if clip else page.screenshot(full_page=True)
                path.write_bytes(data)
                meta = meta_box.input_value()
                print(f"  ID {item['id']:<5} {meta.split('|')[0].strip():<14} -> {path.name}")

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
