# -*- coding: utf-8 -*-
# 工单15：批量截图
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

TEST = Path(__file__).resolve().parent
PAGES = [
    ("跨模态检索测试报告.html", "测试报告截图.png", {"width": 1400, "height": 1200}, True),
    ("_shot_qa.html", "问答运行截图.png", {"width": 1400, "height": 800}, False),
    ("_shot_gallery.html", "图表总览截图.png", {"width": 1400, "height": 1800}, True),
]

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        for html_name, shot, vp, full in PAGES:
            f = TEST / html_name
            if not f.exists(): continue
            page = await browser.new_page(viewport=vp)
            await page.goto(f"file:///{f}", wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(800)
            await page.screenshot(path=str(TEST / shot), full_page=full)
            print(f"已截图 {shot}")
            await page.close()
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
