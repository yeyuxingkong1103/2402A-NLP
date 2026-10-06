# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
"""补充截图：检索接口、来源面板、更多问答。"""

import asyncio
from playwright.async_api import async_playwright
import os

OUTPUT_DIR = r"D:\作业\6-专高NLP 作业\成品\3\测试\screenshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = "http://localhost:8000"

# 检索接口截图
search_tests = [
    ("10-检索-发行股数.png", "武汉力源信息技术股份有限公司本次发行股数是多少"),
    ("11-检索-募集资金.png", "募集资金拟投资哪些项目"),
    ("12-检索-关联方.png", "存在控制关系的关联方"),
]

# 更多问答截图
chat_tests = [
    ("13-问答-力源非控制关联方.png", "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？"),
    ("14-问答-兴图注册资本.png", "武汉兴图新科电子股份有限公司注册资本是多少？"),
    ("15-问答-兴图法定代表人.png", "武汉兴图新科电子股份有限公司法定代表人是谁？"),
    ("16-问答-兴图补充流动资金.png", "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？"),
]


async def take_screenshots():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})

        # 检索接口截图（直接访问 API）
        for filename, query in search_tests:
            print(f"截图：{filename}")
            try:
                await page.goto(f"{BASE_URL}/api/search?query={query}", wait_until="networkidle")
                await page.wait_for_timeout(3000)
                await page.screenshot(path=os.path.join(OUTPUT_DIR, filename), full_page=True)
            except Exception as e:
                print(f"  失败：{e}")

        # 更多问答截图
        for filename, question in chat_tests:
            print(f"截图：{filename}")
            try:
                await page.goto(f"{BASE_URL}/", wait_until="networkidle")
                await page.wait_for_timeout(2000)
                await page.wait_for_selector("#questionInput", timeout=5000)
                await page.fill("#questionInput", question)
                await page.click("#sendBtn")
                await page.wait_for_timeout(8000)
                await page.screenshot(path=os.path.join(OUTPUT_DIR, filename), full_page=True)
            except Exception as e:
                print(f"  失败：{e}")

        await browser.close()

    print(f"\n截图完成！输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(take_screenshots())
