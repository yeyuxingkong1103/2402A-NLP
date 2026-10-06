# -*- coding: utf-8 -*-
"""RAG 真机验证：上传 PDF → 入库 → 带知识库对话 → 校验引用来源。

用法：
    python scripts/verify_rag.py                 # 自动生成中文测试 PDF
    python scripts/verify_rag.py 你的文档.pdf     # 用真实 PDF
"""
import sys
# 解析：命令行参数与退出码
from pathlib import Path
# 解析：路径

import httpx
# 解析：HTTP 客户端

BASE = "http://127.0.0.1:8000"
# 解析：服务地址


def make_test_pdf(path: str) -> None:
    """用 Windows 系统字体生成一份中文测试 PDF（真实字体嵌入，提取正常）。"""
    import fitz
    # 解析：PyMuPDF（延迟导入）

    font_candidates = [
        # 解析：候选中文字体
        "C:/Windows/Fonts/msyh.ttc",
        # 解析：微软雅黑
        "C:/Windows/Fonts/simsun.ttc",
        # 解析：宋体
        "C:/Windows/Fonts/simhei.ttf",
        # 解析：黑体
    ]
    fontfile = next((f for f in font_candidates if Path(f).exists()), None)
    # 解析：取第一个存在的字体

    content = (
        # 解析：测试知识内容
        "高血压健康管理知识\n\n"
        # 解析：标题
        "1. 高血压的诊断标准：在未使用降压药物的情况下，非同日3次测量诊室血压，"
        # 解析：诊断标准
        "收缩压≥140mmHg和（或）舒张压≥90mmHg，可诊断为高血压。\n\n"
        # 解析：标准数值
        "2. 生活方式干预：低盐饮食，每日食盐摄入量不超过5克；适量运动，"
        # 解析：生活方式
        "每周至少150分钟中等强度有氧运动；控制体重；戒烟限酒。\n\n"
        # 解析：运动建议
        "3. 药物治疗：常用降压药物包括钙通道阻滞剂（如氨氯地平）、ACEI类药物"
        # 解析：药物
        "（如依那普利）、ARB类药物（如缬沙坦）、利尿剂（如氢氯噻嗪）等，"
        # 解析：药物举例
        "具体用药方案应遵医嘱。\n\n"
        # 解析：遵医嘱
        "4. 血压监测：家庭自测血压建议早晚各一次，每次测量2~3遍取平均值。\n"
        # 解析：监测建议
    )
    doc = fitz.open()
    # 解析：新建 PDF
    page = doc.new_page()
    # 解析：新建页
    if fontfile:
        # 解析：有中文字体
        page.insert_font(fontname="cjk", fontfile=fontfile)
        # 解析：嵌入字体
        page.insert_text((50, 60), content, fontsize=11, fontname="cjk")
        # 解析：写入中文（真实字体嵌入，提取正常）
    else:
        # 解析：无字体兜底
        page.insert_text((50, 60), content, fontsize=11)
        # 解析：base-14 字体（中文会变点）
    doc.save(path)
    # 解析：保存
    doc.close()
    # 解析：关闭
    print(f"已生成测试 PDF：{path}（字体: {fontfile or 'base-14'}）")
    # 解析：提示


# RAG 真机验证：生成/上传 PDF → 检索对话 → 校验引用来源
def main() -> None:
    # 解析：验证主流程
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "test_knowledge.pdf"
    # 解析：PDF 路径（可传参）
    if not Path(pdf_path).exists():
        # 解析：不存在
        make_test_pdf(pdf_path)
        # 解析：自动生成

    with httpx.Client(base_url=BASE, timeout=300) as c:
        # 解析：HTTP 客户端（300 秒超时——RAG 首次加载模型慢）
        # 登录
        r = c.post("/api/users/login", json={"username": "tester", "password": "test123456"})
        # 解析：尝试登录
        if r.status_code != 200:
            # 解析：未注册过
            r = c.post("/api/users/register", json={"username": "tester", "password": "test123456"})
            # 解析：注册
        r.raise_for_status()
        # 解析：非 200 抛异常
        token = r.json()["token"]
        # 解析：取 token
        headers = {"X-Token": token}
        # 解析：请求头
        print("登录成功")
        # 解析：提示

        # 找林医生
        roles = c.get("/api/roles", headers=headers).json()
        # 解析：角色列表
        doctor = next(x for x in roles if x["name"] == "林医生")
        # 解析：找林医生
        print(f"目标角色：林医生（id={doctor['id']}）")
        # 解析：提示

        # 上传 PDF
        with open(pdf_path, "rb") as f:
            # 解析：读文件
            r = c.post(
                # 解析：上传
                f"/api/knowledge/upload?role_id={doctor['id']}",
                # 解析：知识库接口
                headers=headers,
                # 解析：鉴权头
                files={"file": (Path(pdf_path).name, f, "application/pdf")},
                # 解析：multipart 文件
            )
        if r.status_code != 200:
            # 解析：上传失败
            print("上传失败:", r.status_code, r.text[:300])
            # 解析：打印错误
            sys.exit(1)
            # 解析：退出
        print(f"入库结果：{r.json()}")
        # 解析：打印入库结果

        # 知识库文档列表
        docs = c.get(f"/api/knowledge/docs?role_id={doctor['id']}", headers=headers).json()
        # 解析：查文档列表
        print(f"知识库文档：{docs}")
        # 解析：打印

        # 带 RAG 的对话
        question = "高血压的诊断标准是什么？收缩压超过多少算高血压？"
        # 解析：验证问题
        r = c.post(
            # 解析：对话
            "/api/chat", headers=headers,
            # 解析：接口与鉴权
            json={"role_id": doctor["id"], "content": question, "use_rag": True},
            # 解析：角色、问题、开 RAG
        )
        r.raise_for_status()
        # 解析：非 200 抛异常
        body = r.json()
        # 解析：响应
        print(f"\n问题：{question}")
        # 解析：打印问题
        print(f"林医生回答：{body['reply']}")
        # 解析：打印回答
        print(f"引用来源块数：{len(body['sources'])}")
        # 解析：来源数
        for i, s in enumerate(body["sources"], 1):
            # 解析：逐来源
            print(f"  [来源{i}] {s[:60]}...")
            # 解析：打印前 60 字

        ok = any("140" in s or "90" in s for s in body["sources"])
        # 解析：校验引用含诊断标准数值
        print("\n验证结果：", "通过 ✔（引用来源含诊断标准数值）" if ok else "需人工确认（来源未命中数值）")
        # 解析：结论


if __name__ == "__main__":
    # 解析：入口
    main()
    # 解析：执行
