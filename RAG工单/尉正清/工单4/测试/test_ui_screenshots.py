# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""界面验收截图：用无头浏览器走一遍完整操作流程并截图

用法：python test_ui_screenshots.py [PDF路径]
截图输出到 测试/截图/
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEV = ROOT / "研发"
SHOTS = Path(__file__).resolve().parent / "截图"
PORT = 7860
PDF = sys.argv[1] if len(sys.argv) > 1 else r"D:\BW\RAG 工单\附件\招股说明书1.pdf"
PDF2 = sys.argv[2] if len(sys.argv) > 2 else r"D:\BW\RAG 工单\附件\招股说明书2.pdf"


def shot(page, name):
    SHOTS.mkdir(parents=True, exist_ok=True)
    data = page.screenshot(full_page=True)
    for path in (SHOTS / f"{name}.png", SHOTS / f"{name.split('-')[0]}.png"):
        try:
            path.write_bytes(data)          # 中文文件名在部分环境会写失败，回退用编号
            print(f"  已截图 {path.name}")
            return
        except OSError as exc:                                 # noqa: PERF203
            print(f"  写入 {path.name} 失败（{exc}），尝试备用名")
    print(f"  [警告] {name} 截图保存失败")


def main():
    from playwright.sync_api import sync_playwright

    print("==> 启动服务 ...")
    proc = subprocess.Popen([sys.executable, "-u", "app.py"], cwd=str(DEV),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 1000})

            # 等待服务就绪
            for _ in range(40):
                try:
                    page.goto(f"http://127.0.0.1:{PORT}/", timeout=3000)
                    break
                except Exception:                              # noqa: BLE001
                    time.sleep(2)
            else:
                print("[错误] 服务未能启动"); return 1
            page.wait_for_timeout(3000)

            print("==> 1. 界面首页")
            shot(page, "01-界面首页")

            print("==> 2. 上传 PDF")
            page.locator('input[type="file"]').first.set_input_files([PDF, PDF2])
            page.wait_for_timeout(2000)
            shot(page, "02-已上传PDF")

            print("==> 3. 初始化知识库（等待编码）")
            page.locator('button:has-text("初始化知识库")').first.click()
            for _ in range(60):
                page.wait_for_timeout(2000)
                body = page.inner_text("body")
                if "初始化完成" in body or "缓存加载完成" in body:
                    break
            shot(page, "03-知识库就绪")

            print("==> 4. RAG 模式提问")
            # 状态栏那几个 textarea 是禁用状态的，要挑可用的那个（问题输入框）
            box = page.locator("textarea:enabled").first
            box.fill("报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？")
            page.locator('button:has-text("提问")').first.click()
            page.wait_for_timeout(12000)
            shot(page, "04-RAG回答")

            print("==> 5. 反馈")
            try:
                page.locator('button:has-text("回答正确")').first.click()
                page.wait_for_timeout(1500)
                shot(page, "05-反馈记录")
            except Exception as exc:                           # noqa: BLE001
                print(f"  反馈截图跳过：{exc}")

            print("==> 6. 纯 LLM 对比模式")
            try:
                page.locator('label:has-text("纯 LLM")').first.click()
                box.fill("武汉兴图新科电子股份有限公司法定代表人是谁？")
                page.locator('button:has-text("提问")').first.click()
                page.wait_for_timeout(12000)
                shot(page, "06-纯LLM对比")
            except Exception as exc:                           # noqa: BLE001
                print(f"  对比截图跳过：{exc}")

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
