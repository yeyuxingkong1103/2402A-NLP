# -*- coding: utf-8 -*-
"""命令行演示模块：用四轮问答演示查询改写、答案缓存与按角色切换提示词。"""

import argparse                               # 导入 argparse，用于解析命令行参数
import time                                   # 导入 time，用于统计耗时

import rag                                    # 导入问答模块，复用其中的 ask 链路
import config                                 # 导入配置模块，用于读取集合名等配置
import db_user                                # 导入用户数据模块，用于准备测试用户与会话
import memory                                 # 导入记忆模块，用于打印记忆落地情况
from logger import get_logger                 # 导入日志工具，用于记录演示过程
from prompt import get_system_prompt          # 导入角色提示词查询函数

logger = get_logger("cli")                    # 创建本模块的 logger 实例


def _ensure_test_env(user_id: int, role_id: int, conversation_id: int) -> tuple:
    """确保测试环境就绪：初始化角色、按需创建测试用户与会话，返回可用的三个编号。"""
    db_user.init_default_roles()                     # 幂等初始化两个预置角色
    if not db_user.get_user(user_id):                # 测试用户不存在
        user_id = db_user.register_user("test", "test123456") or user_id   # 注册测试用户
        logger.info("已创建测试用户 test（id=%s）", user_id)   # 记录日志
    if not db_user.get_conversation(conversation_id):   # 测试会话不存在
        conversation_id = db_user.create_conversation(user_id, role_id, "测试会话")   # 建一条
        logger.info("已创建测试会话（id=%s）", conversation_id)   # 记录日志
    return user_id, role_id, conversation_id         # 返回可用的三个编号


def _ensure_general_role(user_id: int) -> int:       # 内部函数：确保通用助手角色存在
    """确保 roles 表里有通用助手角色，返回其编号；用于演示按角色切换提示词。"""
    role = db_user.get_role_by_name("general_assistant")   # 按名称查角色
    if not role:                                     # 不存在则先初始化预置角色再查一次
        db_user.init_default_roles()                 # 幂等插入两个预置角色
        role = db_user.get_role_by_name("general_assistant")   # 再查一次
    return role["role_id"] if role else 1            # 仍取不到则退回 1


def _show(result: dict) -> None:                     # 内部函数：打印一轮问答的关键信息
    """打印一轮问答的改写结果、缓存命中、召回条数、记忆条数与答案片段。"""
    print(f"  rewritten_query = {result['rewritten_query']}")                        # 改写结果
    print(f"  cache_hit = {result['cache_hit']} | 召回 {len(result['contexts'])} 条"    # 缓存与召回
          f" | history={result['history_used']} long_mem={result['long_memory_used']}")   # 记忆条数
    print(f"  answer[:80] = {result['answer'][:80]}")                                 # 答案片段


def main() -> None:                                  # 命令行演示入口
    """命令行入口：四轮问答验证查询改写、缓存与按角色切换提示词。"""
    parser = argparse.ArgumentParser(description="RAG 电力维修问答（命令行演示）")   # 创建解析器
    parser.add_argument("query", help="要提问的问题")   # 位置参数：第一轮问题
    parser.add_argument("follow_up", nargs="?", default="那具体怎么操作？",
                        help="第二轮追问，验证查询改写")   # 可选：第二轮问题
    parser.add_argument("--top_k", type=int, default=None, help="检索条数")   # 可选检索条数
    parser.add_argument("--user-id", type=int, default=1, help="用户编号")   # 测试用户编号
    parser.add_argument("--role-id", type=int, default=1, help="角色编号")   # 测试角色编号
    parser.add_argument("--conversation-id", type=int, default=1, help="会话编号")   # 测试会话编号
    args = parser.parse_args()                       # 解析参数
    start = time.time()                              # 记录耗时起点
    uid, role_id, cid = _ensure_test_env(args.user_id, args.role_id, args.conversation_id)   # 准备环境
    print("=" * 78)                                  # 打印分隔线
    print(f"测试环境：user_id={uid} role_id={role_id} conversation_id={cid}")   # 打印环境
    print("【第一轮】", args.query)                    # 打印小标题与问题
    _show(rag.ask(args.query, uid, role_id, cid, args.top_k))            # 第一轮问答
    print("【第二轮】", args.follow_up)                # 打印小标题与追问
    _show(rag.ask(args.follow_up, uid, role_id, cid, args.top_k))        # 第二轮问答
    print("【第三轮】重复第一轮问题，验证缓存")          # 打印小标题
    _show(rag.ask(args.query, uid, role_id, cid, args.top_k))            # 第三轮问答
    general = _ensure_general_role(uid)              # 确保通用助手角色可用
    print(f"【第四轮】用 general_assistant（role_id={general}）问同一问题")   # 打印小标题
    print(f"  该系统提示词：{get_system_prompt('general_assistant')}")   # 打印已切换的提示词
    _show(rag.ask(args.query, uid, general, cid, args.top_k))            # 第四轮问答
    print("-" * 78)                                  # 打印分隔线
    _print_storage(uid, cid)                         # 打印三处记忆的落地情况
    print(f"总耗时：{time.time() - start:.1f} 秒")     # 打印耗时
    print("=" * 78)                                  # 打印分隔线
    logger.info("四轮问答结束，总耗时 %.1f 秒", time.time() - start)   # 记录耗时


def _print_storage(user_id: int, conversation_id: int) -> None:   # 内部函数：打印存储落地情况
    """打印 MySQL 消息数、Redis 短期记忆条数与 Milvus 长期记忆行数。"""
    messages = db_user.list_messages(conversation_id, limit=200)   # 读会话消息
    print(f"  MySQL messages：{len(messages)} 行")                  # 打印行数
    print(f"  Redis mem:short:{user_id}：{len(memory.get_short_memory(user_id))} 条")   # 打印条数
    client = memory.get_memory_client()              # 取 Milvus 客户端
    name = config.MEMORY_COLLECTION                  # 记忆集合名
    if client.has_collection(name):                  # 集合已存在才统计
        client.load_collection(name)                 # 加载集合
        counted = client.query(collection_name=name, filter="", output_fields=["count(*)"])   # 统计
        print(f"  Milvus {name}：{int(counted[0].get('count(*)', 0)) if counted else 0} 行")   # 打印行数


if __name__ == "__main__":                           # 支持 python -m cli 直接运行
    main()                                           # 执行命令行入口
