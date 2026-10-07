# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Query理解优化任务
"""使用 Playwright 截图：首页 / 多轮对话过程 / 知识库 / API 文档。"""

import os
import sys
import time
from playwright.sync_api import sync_playwright

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
os.makedirs(OUT, exist_ok=True)

BASE = "http://127.0.0.1:8000"


def shoot():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, locale="zh-CN")
        page = ctx.new_page()

        # 1. 首页
        print("[1/5] 首页...")
        page.goto(f"{BASE}/web/index.html", wait_until="networkidle")
        page.wait_for_timeout(1500)
        page.screenshot(path=os.path.join(OUT, "01-首页.png"), full_page=False)

        # 2. 多轮对话 Q1
        print("[2/5] Q1: 兴图军用收入...")
        page.fill("#questionInput", "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？")
        page.click("#sendBtn")
        page.wait_for_timeout(12000)
        page.screenshot(path=os.path.join(OUT, "02-多轮对话Q1.png"), full_page=False)

        # 3. Q2 指代消解
        print("[3/5] Q2: 指代消解...")
        page.fill("#questionInput", "他参与的哪个工程荣获了国家科技进步一等奖？")
        page.click("#sendBtn")
        page.wait_for_timeout(12000)
        page.screenshot(path=os.path.join(OUT, "03-多轮对话Q2-指代消解.png"), full_page=False)

        # 4. Q4 主语切换
        print("[4/5] Q3+Q4...")
        page.fill("#questionInput", "这个公司的法定代表人是谁？")
        page.click("#sendBtn")
        page.wait_for_timeout(12000)
        page.fill("#questionInput", "那武汉力源信息技术股份有限公司呢？")
        page.click("#sendBtn")
        page.wait_for_timeout(12000)
        page.screenshot(path=os.path.join(OUT, "04-多轮对话Q4-主语切换.png"), full_page=False)

        # 5. 知识库
        print("[5/5] 知识库 + API 文档...")
        page.click("button.tab[data-view='kb']")
        page.wait_for_timeout(1500)
        page.screenshot(path=os.path.join(OUT, "05-知识库管理.png"), full_page=False)

        page.goto(f"{BASE}/docs", wait_until="networkidle")
        page.wait_for_timeout(2000)
        page.screenshot(path=os.path.join(OUT, "06-API文档.png"), full_page=False)

        browser.close()
    print("done. saved to:", OUT)


if __name__ == "__main__":
    shoot()
