# -*- coding: utf-8 -*-
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright
TEST = Path(__file__).resolve().parent
PAGES = [
    ("性能测试报告.html", "性能报告截图.png", {"width":1400,"height":1100}, True),
    ("_shot_benchmark.html", "压测日志截图.png", {"width":1400,"height":800}, False),
    ("_shot_gallery.html", "图表总览截图.png", {"width":1400,"height":2200}, True),
]
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=True)
        for h, s, vp, f in PAGES:
            fp = TEST / h
            if not fp.exists(): continue
            pg = await b.new_page(viewport=vp)
            await pg.goto(f"file:///{fp}", wait_until="domcontentloaded", timeout=60000)
            await pg.wait_for_timeout(800)
            await pg.screenshot(path=str(TEST / s), full_page=f)
            print(f"已截图 {s}")
            await pg.close()
        await b.close()
if __name__ == "__main__":
    asyncio.run(main())
