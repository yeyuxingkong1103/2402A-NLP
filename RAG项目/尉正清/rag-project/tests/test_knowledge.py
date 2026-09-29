# tests/test_knowledge.py
"""知识入库与更新接口测试。

以**参数校验与错误分支**为主：真正跑一次全量入库会改动知识库，
不适合放在自动化测试里（那属于运维脚本 scripts/knowledge_base.py 的职责）。

唯一的例外是 `TestUpload`——它上传一份现场合成的小 PDF 并在 teardown 里
按 source 删除，跑完库容复原。留这个例外是因为「上传 → 去水印 → 解析 → 分块
→ 入库」这条链路此前只有人工验证，而它恰恰是最常改动的部分。
"""
import io
import os

import pytest

from tests.conftest import BASE_URL


class TestIngest:

    def test_unknown_role_rejected(self, client):
        r = client.post(BASE_URL + "/api/ingest/dataset",
                        json={"role_key": "no_such_role"}, timeout=30)
        assert r.status_code == 400
        assert "角色不存在" in r.json()["detail"]

    def test_upload_rejects_unsupported_extension(self, client):
        files = {"file": ("test.docx", io.BytesIO(b"dummy"),
                          "application/octet-stream")}
        r = client.post(BASE_URL + "/api/ingest/upload",
                        files=files, data={"role_key": "lawyer"}, timeout=60)
        assert r.status_code == 400
        assert "仅支持" in r.json()["detail"]

    def test_upload_rejects_unknown_role(self, client):
        files = {"file": ("t.txt", io.BytesIO("内容".encode("utf-8")), "text/plain")}
        r = client.post(BASE_URL + "/api/ingest/upload",
                        files=files, data={"role_key": "no_such_role"}, timeout=60)
        assert r.status_code == 400

    def test_upload_rejects_empty_file(self, client):
        files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
        r = client.post(BASE_URL + "/api/ingest/upload",
                        files=files, data={"role_key": "lawyer"}, timeout=60)
        assert r.status_code == 400


class TestUpload:
    """上传并解析入库——自清理用例，见模块说明。"""

    @pytest.mark.slow
    def test_watermarked_pdf_parsed_and_watermark_stripped(self, client):
        """走完整解析链路：切片要落库，水印不能落库。

        标 slow 是因为真实解析耗时数十秒（MinerU ~40s / 兜底视 OCR 页数而定），
        日常回归用 `-m "not slow"` 跳过。
        """
        from app.config import settings
        from app.db.milvus_conn import expr_eq, get_milvus
        from tests.pdf_fixtures import WM_TEXT, watermarked_bytes

        files = {"file": ("wm_upload_test.pdf", watermarked_bytes(),
                          "application/pdf")}
        r = client.post(BASE_URL + "/api/ingest/upload", files=files,
                        data={"role_key": "lawyer"}, timeout=900)
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        source = data["source"]

        try:
            assert data["chunks"] > 0
            rows = get_milvus().query(settings.MILVUS_COLLECTION,
                                      expr_eq(source=source, role_id="lawyer"),
                                      output_fields=["text"], limit=5000)
            assert len(rows) == data["chunks"]
            joined = "\n".join(x.get("text", "") for x in rows)
            assert "正文段落" in joined
            assert WM_TEXT not in joined          # 水印不能进知识库
        finally:
            client.request("DELETE", BASE_URL + "/api/update/document",
                           json={"source": source, "role_key": "lawyer"},
                           timeout=120)
            # 删除接口只清 Milvus 与 Redis，**不删落盘文件**——不收的话
            # 每跑一次测试就在 uploads/ 留一个孤儿文件
            # （见 docs/10-测试报告.md §7.6）。
            try:
                os.remove(os.path.join(settings.UPLOADS_DIR, source))
            except OSError:
                pass


class TestUpdate:

    def test_update_nonexistent_file_rejected(self, client):
        r = client.post(BASE_URL + "/api/update/document",
                        json={"file_path": "/tmp/definitely_not_here.jsonl",
                              "role_key": "lawyer"}, timeout=30)
        assert r.status_code == 400

    def test_update_unknown_role_rejected(self, client):
        r = client.post(BASE_URL + "/api/update/document",
                        json={"file_path": "/tmp/x.jsonl",
                              "role_key": "no_such_role"}, timeout=30)
        assert r.status_code == 400

    def test_update_dataset_unknown_role_rejected(self, client):
        r = client.post(BASE_URL + "/api/update/dataset/no_such_role", timeout=30)
        assert r.status_code == 400

    def test_update_unchanged_file_skips(self, client, tmp_path):
        """内容未变时应跳过重建——用哈希比对实现幂等。"""
        import json
        from app.config import settings

        # 指向一个真实存在的数据集文件；未变更时会命中哈希比对直接返回
        path = "%s/lawyer/legal_qa.jsonl" % settings.DATA_DIR
        r = client.post(BASE_URL + "/api/update/document",
                        json={"file_path": path, "role_key": "lawyer",
                              "force": False}, timeout=300)
        assert r.status_code == 200
        assert r.json()["data"]["changed"] is False
        assert "无需更新" in r.json()["msg"]


class TestDelete:

    def test_delete_nonexistent_source_is_idempotent(self, client):
        """删除不存在的来源不应报错——磁盘与向量库状态可能不同步。"""
        r = client.request("DELETE", BASE_URL + "/api/update/document",
                           json={"source": "not_a_real_file.jsonl",
                                 "role_key": "lawyer"}, timeout=30)
        assert r.status_code == 200

    def test_delete_unknown_role_rejected(self, client):
        r = client.request("DELETE", BASE_URL + "/api/update/document",
                           json={"source": "x.jsonl", "role_key": "nope"},
                           timeout=30)
        assert r.status_code == 400
