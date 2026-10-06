#!/bin/bash
# ============================================================
# 巡检脚本：查看离线知识库入库（ingest）进度与相关组件状态
# 用途：一键查看 ingest 进程是否存活、最近日志、OCR 缓存、已入库文档、Milvus 行数、内存
# 用法：bash scripts/check_ingest.sh
# ============================================================
# 进入项目根目录，保证 data/ logs/ 等相对路径正确
cd /home/dabaie/code/psychologist
# 从 PID 文件读取 ingest 进程号（文件不存在则置空，后续 ps 会因空参数被 2>/dev/null 吞掉报错）
PID=$(cat data/ingest.pid 2>/dev/null)
echo "=== ingest 进程 ==="
# 仅显示该 PID 的 pid/运行时长/内存占用；tail -1 去掉表头行
ps -o pid,etime,rss -p "$PID" 2>/dev/null | tail -1
echo ""
echo "=== 最近 12 行日志 ==="
# 查看日志尾部并截断每行到 130 字符，防止超长行刷屏
tail -n 12 logs/ingest_full.log | cut -c1-130
echo ""
echo "=== OCR 缓存（本轮新写入）==="
# OCR 缓存目录内容（按修改时间排序，tail -6 只看最近的几条）
ls -la data/ocr_cache/ 2>/dev/null | tail -6
# wc -l 统计缓存条目数（不含 ls 的总行）
echo "缓存条目数: $(ls data/ocr_cache/ 2>/dev/null | wc -l)"
echo ""
echo "=== 已入库文档 ==="
# 直连 MySQL 查询知识库文档表：id、角色、标题前 32 字、状态、分块数
mysql -h127.0.0.1 -P3307 -udev -p355359 rag_roleplay --default-character-set=utf8mb4 -e \
 "SELECT d.id, d.persona_id AS pid, LEFT(d.title,32) AS title, d.status, d.chunk_count FROM knowledge_docs d ORDER BY d.id;" 2>/dev/null
echo ""
echo "=== Milvus ==="
# 用项目固定虚拟环境的 Python 直连 Milvus，统计 persona_knowledge 集合的行数（向量条数）
/home/dabaie/code/my_project/.venv/bin/python -c "
from pymilvus import MilvusClient
c = MilvusClient(uri='http://127.0.0.1:19530', timeout=20)
print('persona_knowledge 总行数:', c.get_collection_stats('persona_knowledge').get('row_count'))
" 2>&1 | tail -2
echo ""
# 查看系统内存概况（total/used），辅助判断大模型/向量库是否吃满内存
free -m | head -2