"""S3 清洗：把 MinerU 的 content_list.json 变成可下游消费的 blocks.jsonl。

模块划分（对应用户要求的四类处置）：
    rules_drop      剔除类
    rules_protect   保护类
    rules_convert   转换类
    rules_repair    修复类
    report          四类分组报告
    loader/paths/io_utils   输入输出
    pipeline        编排

对外唯一入口是 backend/clean_parsed.py；本包不被单独执行。
"""

CLEAN_RULE_VERSION = "clean-v1"
