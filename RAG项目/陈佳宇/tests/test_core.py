import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEEPSEEK_API_KEY, MYSQL_HOST, REDIS_HOST, JWT_SECRET, HISTORY_MAX_TURNS
from db import hash_password, verify_password
from auth import create_token, verify_token


def test_config_loaded():
    """测试配置读取"""
    assert DEEPSEEK_API_KEY != "", "DeepSeek API Key不能为空"
    assert MYSQL_HOST != "", "MySQL Host不能为空"
    assert REDIS_HOST != "", "Redis Host不能为空"
    assert JWT_SECRET != "", "JWT Secret不能为空"
    assert HISTORY_MAX_TURNS > 0, "历史轮数必须大于0"
    print("✅ 配置读取测试通过")


def test_password_hash():
    """测试密码加密"""
    pwd = "test123456"
    hashed = hash_password(pwd)
    assert ":" in hashed, "加密密码格式应为 盐值:哈希"
    assert hashed != pwd, "加密后不能等于明文"
    # 两次加密结果不同（因为盐值随机）
    hashed2 = hash_password(pwd)
    assert hashed != hashed2, "两次加密盐值应不同"
    print("✅ 密码加密测试通过")


def test_password_verify():
    """测试密码验证"""
    pwd = "test123456"
    hashed = hash_password(pwd)
    # 正确密码
    assert verify_password(pwd, hashed) == True, "正确密码应验证通过"
    # 错误密码
    assert verify_password("wrong", hashed) == False, "错误密码应验证失败"
    # 兼容旧版明文
    assert verify_password("123456", "123456") == True, "旧版明文密码应兼容"
    print("✅ 密码验证测试通过")


def test_jwt_token():
    """测试JWT Token生成和验证"""
    username = "testuser"
    token = create_token(username)
    assert token != "", "Token不能为空"
    # 验证token
    result = verify_token(token)
    assert result == username, f"验证后用户名应为{username}"
    # 错误token
    assert verify_token("invalid.token.here") is None, "无效Token应返回None"
    print("✅ JWT Token测试通过")


def test_query_rewrite():
    """测试Query改写"""
    from rag_core import query_rewrite
    # 包含关键词的应该被扩写
    result = query_rewrite("Alice多大？")
    assert "年龄" in result or "多少岁" in result, "多大应被扩写"
    # 不包含关键词的原样返回
    result = query_rewrite("你好")
    assert result == "你好", "无关键词应原样返回"
    print("✅ Query改写测试通过")


def test_post_process():
    """测试后处理"""
    from rag_core import post_process
    # 去除重复的角色名前缀
    result = post_process("Alice：Alice：你好", "Alice")
    assert result == "你好", "应去除重复的Alice前缀"
    # 正常回答不变
    result = post_process("你好", "Alice")
    assert result == "你好", "正常回答不应改变"
    print("✅ 后处理测试通过")


if __name__ == "__main__":
    test_config_loaded()
    test_password_hash()
    test_password_verify()
    test_jwt_token()
    test_query_rewrite()
    test_post_process()
    print("\n🎉 所有单元测试通过！")
