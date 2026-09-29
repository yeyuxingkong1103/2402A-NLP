"""会话摘要的触发阈值与长度上限（单点调参，自 summary_service.py 拆出便于测试注入）。

背景：短期记忆是"滑动窗口 + 硬截断"——消息列表恒定为 max_messages 条
（装配侧为 20 条 = 10 轮问答），第 11 轮起最早的对话被 ltrim 永久丢弃。
会话摘要把"将被丢弃的更早轮次"压成一段短文本，是补这个洞的唯一手段。

调参只改本文件；本模块不 import 任何东西（纯常量），可被任意层安全引用。
"""

# 一轮问答在窗口里占 2 条消息（user 提问 + assistant 回答）
MESSAGES_PER_TURN = 2

# 节流：距上次摘要又累积满 6 条消息（3 轮）才再摘要一次。
# 为什么不每轮都摘要：窗口满之后每一轮都会淘汰旧消息，不节流就是每轮一次 LLM 调用；
# 3 轮一次把成本压到 1/3，代价是摘要最多滞后 3 轮（仍在最近 6 条原文覆盖范围内）。
SUMMARY_MIN_NEW_MESSAGES = 6
SUMMARY_MIN_NEW_TURNS = SUMMARY_MIN_NEW_MESSAGES // MESSAGES_PER_TURN

# 最近 6 条（3 轮）原文不参与摘要输入：原文比摘要保真，这段本来还留在窗口里，
# 交给改写器/模型用原文即可，压成摘要反而丢信息。
SUMMARY_KEEP_RECENT_MESSAGES = 6

# 摘要硬上限（字符）。约合 450 tokens，超过则截断并在日志里记一次 warning。
# 上限存在的意义：摘要会被注入每一轮提示词，必须防止它无限膨胀吃掉法源预算。
SUMMARY_MAX_CHARS = 300

# 单条消息进摘要输入时的截断长度：一次问答的 assistant 回答可达数千字，
# 全量拼进摘要 prompt 会让输入 token 失控（方案估算为 6 条 ≈ 1200 字）。
SUMMARY_INPUT_MESSAGE_CHARS = 400
