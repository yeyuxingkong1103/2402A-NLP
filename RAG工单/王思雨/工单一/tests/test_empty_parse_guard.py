# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 本测试职责：钉住「扫描件静默入库 0 条」这一缺陷，防止它从任何入口复发。
#
# 为什么值得一条回归测试，而不是手工修一次就完：
#   网页上传、直接调 API、批量导入脚本是三条不同的入口。当初只在 parse_pdf 里手写了一句
#   「解析为空就报错」，换个入口（绕过 UI 直接调 API、或跑 scripts/build_index.py）就会静默
#   写入 0 条。静默成功比显式失败危险得多 —— 用户拿到一个空知识库，还以为已经建好了。
#   因此这里分层钉住：解析层抛错、写入层拒绝、上传任务置 failed、HTTP 接口如实上报。
#
# 所有测试用 pytest 的 tmp_path 构造临时 PDF，跑完即随临时目录销毁，不在仓库里留模拟数据文件。

import pymupdf                    # 构造测试用 PDF
import pytest                     # 测试框架
from pathlib import Path          # 路径处理

from app import rag, store        # 被测的解析层与存储层


def _make_textless_pdf(path: Path) -> Path:
    """造一个无文本层的单页 PDF（只有矢量图形，不含任何文字），用来模拟扫描件。"""
    doc = pymupdf.open()                                                          # 新建文档
    page = doc.new_page()                                                         # 新建一页
    page.draw_rect(pymupdf.Rect(50, 50, 300, 200), color=(0, 0, 0), fill=(0.85, 0.85, 0.85))  # 画个方块
    doc.save(str(path))                                                           # 存盘
    doc.close()                                                                   # 释放句柄
    return path                                                                   # 返回路径


def _make_text_pdf(path: Path) -> Path:
    """造一个含文本层的单页 PDF，作为对照组，用来证明闸门不会误伤正常文件。"""
    doc = pymupdf.open()                                                          # 新建文档
    page = doc.new_page()                                                         # 新建一页
    # 用 ASCII 文案：pymupdf 内置字体不含中文字形，插中文会渲染成空白，导致对照组失真
    page.insert_text((72, 72), "The legal representative of the issuer is Cheng Jiaming.")   # 写一行字
    doc.save(str(path))                                                           # 存盘
    doc.close()                                                                   # 释放句柄
    return path                                                                   # 返回路径


def test_textless_pdf_really_has_no_text(tmp_path):
    """前置假设自检：确认我们造的"扫描件"确实提取不出任何文本，否则后面的测试是假的。"""
    p = _make_textless_pdf(tmp_path / "scan.pdf")                                 # 造扫描件
    doc = pymupdf.open(str(p))                                                    # 打开
    assert doc[0].get_text().strip() == ""                                        # 文本层必须为空
    doc.close()                                                                   # 释放句柄


def test_control_pdf_with_text_is_accepted(tmp_path):
    """对照组：有文本层的 PDF 必须正常解析出块 —— 证明闸门不会误伤正常文件。"""
    p = _make_text_pdf(tmp_path / "normal.pdf")                                   # 造正常 PDF
    blocks, parser = rag.parse_pdf(p, use_mineru=False)                           # 走快速解析路径
    assert parser == "fast"                                                       # 命中降级解析器
    assert blocks, "有文本层的 PDF 不应被判定为空"                                   # 必须解析出内容


def test_scanned_pdf_raises_instead_of_returning_empty(tmp_path):
    """解析层：无文本层时必须显式抛错，而不是返回空块列表让上层"成功"入库。"""
    p = _make_textless_pdf(tmp_path / "scan.pdf")                                 # 造扫描件
    with pytest.raises(RuntimeError) as ei:                                       # 期望抛错
        rag.parse_pdf(p, use_mineru=False)                                        # 解析
    msg = str(ei.value)                                                           # 错误信息
    assert "未提取到任何文本" in msg or "扫描件" in msg                             # 且提示可操作


def test_index_chunks_rejects_empty_before_touching_storage(monkeypatch):
    """写入层：空分块必须在**删除旧数据之前**就被拒绝，且不得触碰存储层。

    原来的实现顺序是「先按 doc_id 删除旧数据、再写入」，
    所以空的解析结果不仅会静默成功，还会顺手把该文档已有的索引清掉。
    """
    touched = []                                                                  # 记录存储层是否被调用

    def _boom():                                                                  # 替身：一被调用就说明顺序错了
        touched.append(True)                                                      # 留痕
        raise AssertionError("空分块不应触及存储层（说明删除发生在校验之前）")           # 直接失败

    monkeypatch.setattr(store, "ensure_collection", _boom)                        # 替换存储层入口
    with pytest.raises(ValueError):                                               # 期望 ValueError
        store.index_chunks([], "doc_scanned", "scan.pdf")                         # 空分块入库
    assert touched == []                                                          # 校验必须发生在删除之前


def test_build_index_task_marks_failed(tmp_path):
    """上传后台任务：走 _build_index 真实代码路径，任务状态必须是 failed，而不是 done/0 条。"""
    from app import main                                                          # 延迟导入，避免影响其它测试

    p = _make_textless_pdf(tmp_path / "scan.pdf")                                 # 造扫描件
    tid = "pytest-scan-guard"                                                     # 固定任务号
    main._TASKS[tid] = {"status": "queued", "message": "排队中", "file": p.name}   # 初始化任务状态
    try:                                                                          # 保证清理
        main._build_index(p, tid, use_mineru=False)                               # 执行后台建索引
        st = main._TASKS[tid]                                                     # 取回状态
    finally:                                                                      # 无论成败
        main._TASKS.pop(tid, None)                                                # 清理任务表，不污染后续
    assert st["status"] == "failed", f"扫描件竟被当成成功：{st}"                      # 必须失败
    assert "未提取到任何文本" in st["message"] or "扫描件" in st["message"]          # 原因可读


def test_upload_endpoint_reports_failure(tmp_path, monkeypatch):
    """HTTP 层（最贴近"用户上传"的一句话断言）：传扫描件 → 任务状态为 failed。"""
    from fastapi.testclient import TestClient                                     # 测试客户端
    from app import config as C, main                                             # 配置与应用

    monkeypatch.setattr(C, "UPLOAD_DIR", tmp_path)                                # 别往真实 data/uploads 写测试文件
    p = _make_textless_pdf(tmp_path / "scan.pdf")                                 # 造扫描件
    client = TestClient(main.app)                                                 # 不进入 lifespan：测试不需要预热模型
    with open(p, "rb") as fh:                                                     # 以文件流上传
        r = client.post("/api/kb/upload",                                         # 调上传接口
                        files={"file": ("scan.pdf", fh, "application/pdf")},      # 文件字段
                        data={"use_mineru": "false"})                             # 跳过 MinerU，走快速解析
    assert r.status_code == 200, r.text                                           # 上传本身受理
    tid = r.json()["task_id"]                                                     # 取任务号
    st = client.get(f"/api/kb/task/{tid}").json()                                 # 轮询任务状态
    assert st["status"] == "failed", f"接口层竟报成功：{st}"                        # 必须是失败，不能是 done
