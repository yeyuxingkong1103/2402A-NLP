import mysql.connector
from dotenv import load_dotenv
import os

load_dotenv()

conn = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DB")
)
cursor = conn.cursor()

# 角色表
sql_role = """
CREATE TABLE IF NOT EXISTS `role` (
  id INT AUTO_INCREMENT PRIMARY KEY COMMENT '角色ID',
  role_name VARCHAR(100) NOT NULL COMMENT '角色名称',
  role_desc TEXT COMMENT '角色人设描述',
  system_prompt TEXT COMMENT '角色system提示词模板',
  milvus_collection VARCHAR(100) COMMENT '绑定Milvus知识库集合名',
  create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
  update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""
cursor.execute(sql_role)

# 对话记录表（可选，备用）
sql_chat = """
CREATE TABLE IF NOT EXISTS `chat_history` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  role_id INT NOT NULL,
  session_id VARCHAR(128) NOT NULL,
  user_query TEXT,
  bot_resp TEXT,
  create_time DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""
cursor.execute(sql_chat)

conn.commit()
print("✅ 数据表创建完成")
cursor.close()
conn.close()
