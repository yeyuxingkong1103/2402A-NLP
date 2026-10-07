# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
"""Playwright 截图：首页/问答过程/知识库/API文档/检索配置。"""
import time
from playwright.sync_api import sync_playwright

OUT = r"D:\作业\6-专高NLP 作业\成品\7\测试"
URL = "http://127.0.0.1:8000"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})

    # 1. 首页
    page.goto(f"{URL}/web/index.html")
    page.wait_for_timeout(2000)
    page.screenshot(path=f"{OUT}\\01-首页.png", full_page=False)
    print("01-首页.png")

    # 2. 问答过程
    page.fill("#questionInput", "平安银行2019年的董事长是谁？")
    page.click("#sendBtn")
    page.wait_for_timeout(8000)
    page.screenshot(path=f"{OUT}\\02-问答过程.png", full_page=False)
    print("02-问答过程.png")

    # 3. 知识库管理
    page.click("text=知识库")
    page.wait_for_timeout(2000)
    page.screenshot(path=f"{OUT}\\03-知识库管理.png", full_page=False)
    print("03-知识库管理.png")

    # 4. API 文档
    page.goto(f"{URL}/docs")
    page.wait_for_timeout(2000)
    page.screenshot(path=f"{OUT}\\04-API文档.png", full_page=False)
    print("04-API文档.png")

    # 5. 检索配置接口
    page.goto(f"{URL}/docs#/检索策略")
    page.wait_for_timeout(1500)
    page.screenshot(path=f"{OUT}\\05-检索配置接口.png", full_page=False)
    print("05-检索配置接口.png")

    # 6. 检索配置面板（前端）
    page.goto(f"{URL}/web/index.html")
    page.wait_for_timeout(1000)
    page.click("text=检索配置")
    page.wait_for_timeout(1500)
    page.screenshot(path=f"{OUT}\\06-检索配置面板.png", full_page=False)
    print("06-检索配置面板.png")

    browser.close()

print("截图完成！")
