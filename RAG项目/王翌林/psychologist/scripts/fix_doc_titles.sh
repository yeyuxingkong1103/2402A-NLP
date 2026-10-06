#!/bin/bash
# 校正知识库文档标题
#   用途：扫描件 PDF 的内置元数据标题常为 "SSReader Print."、"Untitled" 等无效值，
#         文档入库时若已写入这类标题，用本脚本按文件名回退校正（带尾部控制字符的标题一并清理）。
#   用法：
#     bash scripts/fix_doc_titles.sh          # 立即校正
#     bash scripts/fix_doc_titles.sh --wait   # 等待正在运行的导入结束后再校正（读 data/ingest.pid）
# set -u 只启用"未定义变量即报错"，不启用 -e：本脚本重在巡检校正，
# mysql 命令可能因数据问题返回非零，若 -e 会提前中断，故不开启
set -u
# 进入项目根目录，保证 data/ingest.pid 与 mysql 目标库的相对约定正确
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# 统一封装 MySQL 连接参数，避免多处重复写主机/端口/账号/字符集
MYSQL="mysql -h127.0.0.1 -P3307 -udev -p355359 rag_roleplay --default-character-set=utf8mb4"

# --wait 模式：等正在运行的 ingest 导入结束再校正，避免与写库竞争
if [ "${1:-}" = "--wait" ]; then
  PID_FILE=data/ingest.pid
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    # kill -0 只探测不发送信号，确认导入进程是否真的还在
    if kill -0 "$PID" 2>/dev/null; then
      echo "[*] 等待导入进程 $PID 结束 ..."
      # 每 30 秒探测一次，直到进程退出（进程结束后循环自动结束）
      while kill -0 "$PID" 2>/dev/null; do sleep 30; done
    fi
  fi
  echo "[*] 导入已结束 $(date '+%Y-%m-%d %H:%M:%S')"
fi

echo "[*] 开始校正标题 $(date '+%Y-%m-%d %H:%M:%S')"

# 用 heredoc 把两段 SQL 一次性喂给 mysql（'SQL' 引号防止 shell 展开 $ 等字符）
$MYSQL 2>/dev/null <<'SQL'
-- 1) 清理标题中的控制字符（\x00 / 制表符 / 回车）
UPDATE knowledge_docs
SET title = REPLACE(REPLACE(REPLACE(title, CHAR(0), ''), CHAR(9), ''), CHAR(13), '')
WHERE title LIKE CONCAT('%', CHAR(0), '%')
   OR title LIKE CONCAT('%', CHAR(9), '%')
   OR title LIKE CONCAT('%', CHAR(13), '%');

-- 2) 无效元数据标题回退为文件名（去扩展名）
UPDATE knowledge_docs
SET title = SUBSTRING_INDEX(SUBSTRING_INDEX(file_path, '/', -1), '.', 1)
WHERE title REGEXP '^(SSReader|Untitled|Microsoft Word|Document[0-9]|CamScanner|扫描全能王|Image)'
   OR CHAR_LENGTH(TRIM(title)) < 3;
SQL

# 校正后再查一遍结果，人工确认标题是否符合预期
echo "[*] 校正后标题一览："
$MYSQL -e "SELECT id, persona_id, LEFT(title, 46) AS title, chunk_count FROM knowledge_docs ORDER BY persona_id, id;" 2>/dev/null
echo "[*] 完成 $(date '+%Y-%m-%d %H:%M:%S')"