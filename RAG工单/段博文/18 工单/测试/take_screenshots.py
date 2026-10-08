# -*- coding: utf-8 -*-
"""Playwright 截图：质检报告、工作流演示、API 演示。"""
from pathlib import Path

from playwright.sync_api import sync_playwright

TEST = Path(__file__).resolve().parent

targets = [
    ("文档质检报告.html", "文档质检报告截图.png"),
    ("workflow_demo.html", "工作流路由演示截图.png"),
    ("api_demo.html", "API调用演示截图.png"),
]

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    for src, out in targets:
        url = (TEST / src).as_uri()
        page.goto(url, wait_until="networkidle")
        page.screenshot(path=str(TEST / out), full_page=True)
        print("已截图", out)
    browser.close()
print("完成")
