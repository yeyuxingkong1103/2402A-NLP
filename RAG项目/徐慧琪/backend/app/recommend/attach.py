"""附加区块组装。

存在的理由：把「案由 → 律师 → 费用」三块收成一个纯组装点，供问答编排层
一次调用。它是**故障降级**的落点：费用那块挂掉（检索/模型不可用）时标成
unavailable 并继续返回，绝不把服务故障说成「暂无费用口径依据」——
那是 ③a 定下的红线。
"""
from __future__ import annotations

import logging

from app.recommend import cause as cause_mod
from app.recommend import fees, lawyers

# 本模块的故障信号出口。留痕失败与费用故障都只在这里留一条 warning：主流程
# 照旧（用户体验不变），但「连续发生」这件事从「全链路全绿」变成可在日志里看见
logger = logging.getLogger(__name__)

# 默认给三张卡片：FR-9.3 要求「按领域匹配展示」，1~3 张是技术方案 6.4 定的范围
DEFAULT_CARD_COUNT = 3


def _unavailable(exc: Exception) -> dict:
    """故障态的 fee 结果 —— 与 no_corpus 严格区分（渲染文案不同）。

    **必须与 estimate 的每一条分支同形（9 键，含 `unit` 与 `charge_basis`）**：
    渲染方按固定字段读 `fee["unit"]` 等键，少一个键就在**故障这一支**上抛
    KeyError —— 而故障恰恰是最没人手工测到的分支。下面的字段名是**手写镜像**
    estimate 的字段集（不是从 estimate 那侧取来的）：改一边必须同步另一边
    （形状一致性有测试钉住）。`reason` 带上原始异常文本：故障要能被人排查，
    只给一句「暂时不可用」等于把现场丢了。
    """
    return {"status": "unavailable", "low": None, "high": None, "unit": None,
            "charge_basis": None, "basis": None, "source_doc": None,
            "source_no": None, "reason": f"费用信息暂时不可用：{exc}"}


def build(question: str, *, search_fn, generate_fn, log_fn, rng=None) -> dict:
    """组装附加区块。三个依赖全部注入（见 Task 4/5 的接口）。

    留痕（log_fn）失败**吞掉**：审计是旁路，不能让用户因此看不到费用信息；
    检索与生成的失败则转成 unavailable —— 两种失败的归属不同，判别方式是
    「异常在哪个边界被吃掉」：留痕的异常在 guarded_log 内部就结束，传不进下面
    那个 except；检索/生成的异常由 estimate 原样抛出（estimate 内不做任何
    try/except，见其文档字符串），只能落进 except 转成 unavailable。

    降级边界**只圈住 fees.estimate**：cause.tag / cause.field_of /
    lawyers.recommend 是纯计算（不碰外部服务），有意不套 try —— 它们若抛就是
    真 bug，应当暴露给调用方，而不是被这里吞成一次静默降级。
    """
    # 案由只做标签与领域映射，不参与费用检索：认不出时 tag 给 None，
    # 交给 field_of 退到「通用」（卡片照给，见设计 §七），不在这里抛错
    cause_name = cause_mod.tag(question)
    field = cause_mod.field_of(cause_name)
    cards = lawyers.recommend(field, DEFAULT_CARD_COUNT, rng)

    def guarded_log(**fields):
        """把留痕失败挡在费用流程之外；kwargs **原样转发**给 log_fn。

        原样转发是硬要求：`fee_log.record` 对不认识的字段名**静默丢弃**
        （AC-19 的第二道保证：多传的字段没有列接得住），所以组装层一旦改名或
        重排，留痕不会报错、只会变成一格 NULL —— 证据当场作废且没有红灯。
        故这里只加一层异常处理，字段一个都不碰。
        """
        try:
            log_fn(**fields)
        except Exception as exc:  # noqa: BLE001 —— 留痕失败不改变用户体验
            # 用户流程照旧不受影响，但**不许无声**（Task 6/7 交接单点名、Task 9
            # 漏做的那条）：AC-21 的证据链在 MySQL 里，写不进去时没有任何红灯 ——
            # 连续几小时零留痕的表现与「本来就没有问答」完全一样。catch 在这里而不是
            # 把 log_fn 包成一个更早的适配器，是为了让「哪些异常算留痕的」与「哪些算
            # 费用的」在同一个地方可读。fields 里只有案由/片段 id/区间/模型/请求 id
            # 与状态（AC-19：不含用户问句），可原样进日志
            logger.warning("费用留痕写入失败，AC-21 证据丢失：%s；异常：%r",
                           fields, exc)

    # 案由为 None 时喂空串：estimate 的 cause 形参声明为 str，真检索会拿它拼
    # 查询条件、留痕也会照写，透传 None 会在更深处炸成另一种故障（而认不出案由
    # 本不是故障）
    try:
        fee = fees.estimate(cause_name or "", search_fn=search_fn,
                            generate_fn=generate_fn, log_fn=guarded_log)
    except Exception as exc:  # noqa: BLE001 —— 检索/生成故障：降级但不伪装
        fee = _unavailable(exc)

    return {"cause": cause_name, "field": field, "lawyers": cards,
            "fee": fee, "disclaimer": fees.FEE_DISCLAIMER}
