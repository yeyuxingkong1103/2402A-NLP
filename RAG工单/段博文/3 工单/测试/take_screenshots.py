# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
"""截图脚本：自动截图 RAG 系统的关键页面。"""

import asyncio
from playwright.async_api import async_playwright
import os

OUTPUT_DIR = r"D:\作业\6-专高NLP 作业\成品\3\测试\screenshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = "http://localhost:8000"

# 截图配置：(文件名, URL, 等待时间, 操作)
screenshots = [
    ("01-首页.png", f"{BASE_URL}/", 2, None),
    ("02-API文档.png", f"{BASE_URL}/docs", 3, None),
    ("03-健康检查.png", f"{BASE_URL}/api/health", 2, None),
    ("04-系统统计.png", f"{BASE_URL}/api/stats", 2, None),
]

# 问答测试用例
chat_tests = [
    ("05-问答-力源发行股数.png", "武汉力源信息技术股份有限公司本次发行股数是多少？"),
    ("06-问答-力源募集资金.png", "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？"),
    ("07-问答-力源关联方.png", "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁？"),
    ("08-问答-兴图军用收入.png", "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"),
    ("09-问答-兴图技术标准.png", "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"),
]


async def take_screenshots():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})

        # 基础页面截图
        for filename, url, wait_time, _ in screenshots:
            print(f"截图：{filename}")
            try:
                await page.goto(url, wait_until="networkidle")
                await page.wait_for_timeout(wait_time * 1000)
                await page.screenshot(path=os.path.join(OUTPUT_DIR, filename), full_page=True)
            except Exception as e:
                print(f"  失败：{e}")

        # 问答截图（需要先访问首页，然后执行搜索）
        for filename, question in chat_tests:
            print(f"截图：{filename}")
            try:
                await page.goto(f"{BASE_URL}/", wait_until="networkidle")
                await page.wait_for_timeout(2000)

                # 查找输入框并输入问题
                await page.wait_for_selector("#questionInput", timeout=5000)
                await page.fill("#questionInput", question)

                # 点击发送按钮
                await page.click("#sendBtn")

                # 等待结果加载
                await page.wait_for_timeout(8000)

                await page.screenshot(path=os.path.join(OUTPUT_DIR, filename), full_page=True)
            except Exception as e:
                print(f"  失败：{e}")

        await browser.close()

    print(f"\n截图完成！输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(take_screenshots())
