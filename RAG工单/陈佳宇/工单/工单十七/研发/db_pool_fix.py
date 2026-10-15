# -*- coding: utf-8 -*-
# db_pool_fix.py 工单17 数据库、向量库连接池封装，防止连接泄漏
"""
关键点：无论正常返回还是异常抛出，都必须归还连接，使用try‑finally
"""

class SafePoolClient:
    def __init__(self, pool):
        self.pool = pool

    def get_conn(self):
        return self.pool.get()

    def execute_with_conn(self, func, *args,**kwargs):
        """安全执行：获取连接，无论异常与否，归还连接池"""
        conn = None
        try:
            conn = self.get_conn()
            return func(conn, *args,**kwargs)
        finally:
            if conn is not None:
                # 无论成功/报错，都归还连接
                self.pool.put(conn)


# 使用示例
"""
pool = SomeConnectionPool(max_connections=10)
safe_client = SafePoolClient(pool)

def query_func(conn, param):
    return conn.query(param)

# 调用，自动保证归还连接
result = safe_client.execute_with_conn(query_func, "search‑text")
"""
