#!/usr/bin/env python3
"""给评测账号**养出长期记忆**（召回 A/B 的前置步骤）。

为什么必须有这一步（2026-09-25 的教训）：召回是**按用户维度**取的。
拿一个全新账号去跑"召回关 vs 召回开"，两臂的召回结果**都是空**，
于是两臂逐字相同 —— 看起来"没有任何影响"，其实是**测了个寂寞**。

做法：注册（或登录）评测账号 → 在同一会话里发 ``--turns`` 轮**与该用户本人相关的背景对话**
（姓名/所在地/自己的案子/手里的证据），靠 Redis 滑窗把最早几组挤进 Milvus。
谈话内容刻意**不是评测题**，否则等于把答案提前喂给它，A/B 就废了。

用法（云端，链路已起）::

    python scripts/seed_eval_memory.py --username eval-recall-ab --turns 10
    # D5「换性质不同的历史再验一轮」：每份历史用**各自的账号**（记忆按用户维度隔离）
    python scripts/seed_eval_memory.py --username eval-recall-st --flavor statute
    python scripts/seed_eval_memory.py --username eval-recall-dl --flavor daily

⚠️ 每份 ``--flavor`` 必须配**不同账号**（``--username``）：召回按用户维度取，
同一个账号灌第二份历史会把两份记忆混在一起，A/B 的因果就不干净了。
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: 「这位用户自己的背景」——刻意与评测题集**不重合**，只提供"历史噪声 + 本人事实"。
SEED_TURNS = [
    "我叫周明远，住在杭州，最近在跟公司打劳动仲裁",
    "公司拖欠我两个月工资，我先去仲裁委立了案",
    "我的劳动合同上没写清楚工资构成，只有一句按月薪发放",
    "我手里有打卡记录和和主管的微信聊天记录",
    "我老婆也在同一家公司，她的情况跟我差不多",
    "公司说要对我的仲裁申请提反诉，我有点担心",
    "我想知道这种案子一般要多久才能有结果",
    "我还有个朋友问，租房押金不退能不能直接起诉",
    "先谢谢你了，我整理好材料再来问细节",
    "我会把所有证据按时间顺序整理一份清单",
]

#: 「用户自己贴过一段法条/合同条款」——D5 要验的**高风险**历史：记忆里出现了**用户贴的法律文字**，
#: 一旦被模型当成依据抄进答案，就是"把用户的话当法条"。
#:
#: ⚠️ **为什么不是专利法**（2026-09-27 差点踩到的坑）：
#: 评测集里有 **N22「发明专利的保护期是多少年」，claims=('二十年',)，notes 就写着"专利法第四十二条"**。
#: 第一版选的正是这一条（"发明专利权的期限为二十年…"）⇒ 等于**把标准答案喂进长期记忆**：
#: 召回开那一臂可能直接从记忆里答对 N22，看起来像"召回提升了准确率"，其实是**实验被污染**。
#: 所以选题规则是**三条**（写成测试了，见 `tests/test_seed_eval_memory.py`）：
#:   1) 与任何评测题的**问题**没有直接子串重合；
#:   2) 不含任何评测题的 **claim**（长度 >= 3 的，如 二十年 / 五十年 / 十五年 / 未经许可）；
#:   3) 领域关键词在评测集里**零命中**（海商/共同海损/承运人/提单/理货单 = 全 0）。
#: 选定的海事领域满足三条；它**不在**评测题的射程内，所以"答案里出现它"这件事本身就可疑。
STATUTE_TURNS = [
    "我叫陆则言，在宁波做国际货运代理，主要接海运整箱的活儿",
    "上个月我们一条柜子在海上遇到大风，货损了，船公司说这算共同海损，要我们分摊",
    "我把合同里那段抄给你：因共同海损引起的牺牲和费用，由各受益方按各自获救财产的价值比例分摊。",
    "我理解的是「谁受益谁分摊」，可我们这票货其实没救回来多少",
    "船公司还提到「承运人责任期间」，说货到卸货港以后就不归他们管了",
    "我们这条柜子是从宁波出，到汉堡卸，中间在新加坡转了一次船",
    "货主现在要我赔，我这边能找船公司追吗",
    "我住在宁波北仑区，平时主要走欧洲线",
    "先谢谢你了，我把提单和保单找出来再问你",
    "我会把事故报告和理货单都归档留好",
]

#: 「与法律无关的日常」——对照组：记忆与法律问答**毫无关系**。
#: 若这份历史也能改变答案，说明改动来自"多了一段上下文"本身，而不是记忆内容。
DAILY_TURNS = [
    "我叫苏晴，在成都做插画，平时接一些自由职业的单子",
    "最近成都一直下雨，我周末本来想去青城山结果没去成",
    "我在学做川菜，昨天试了回锅肉，火候还是掌握不好",
    "我养了一只橘猫叫团子，它最近老爱趴在键盘上",
    "下个月我打算去云南玩一周，正在看机票",
    "我最近在追一部讲做饭的纪录片，看着看着就饿了",
    "我妈让我过年回家相亲，我有点不想去",
    "我报了个游泳班，教练说我的换气节奏太急",
    "先谢谢你了，下次再聊",
    "我把这些都记在备忘录里了",
]

#: ``--flavor`` → (给操作者看的一句话说明, 轮次表)。
#:
#: 为什么要"味道"：D5 的验收要求**换 1–2 份性质不同的历史**各跑一轮 A/B，
#: 因为只用一份历史（labor）得到的结论只能说"在那一份历史下如何"。
SEED_FLAVORS: dict[str, tuple[str, list[str]]] = {
    "labor": ("劳动仲裁背景（原版；与评测题集不重合的历史噪声）", SEED_TURNS),
    "statute": ("用户自己贴过一条法条原文（**法律味最重**，最可能诱发回音/串味）", STATUTE_TURNS),
    "daily": ("与法律无关的日常闲聊（对照组：记忆与法律问答无关）", DAILY_TURNS),
}



def _post(base: str, path: str, payload: dict, cookie: str = "", timeout: float = 300.0):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"})
    if cookie:
        req.add_header("Cookie", cookie)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", "replace")), \
                resp.headers.get("Set-Cookie", "")
    except urllib.error.HTTPError as exc:
        return exc.code, {"_error": exc.read().decode("utf-8", "replace")[:200]}, ""


def _longterm_chunks(base: str) -> int:
    """读 `/health` 里长期记忆的条数（读不到就抛 —— 不许把"读不到"当成 0）。"""
    with urllib.request.urlopen(base + "/health", timeout=30) as resp:
        health = json.loads(resp.read().decode("utf-8", "replace"))
    return int(((health or {}).get("longterm") or {}).get("chunks") or 0)


def login(base: str, username: str, password: str, timeout: float) -> tuple[str, str]:
    """注册或登录，返回 ``(user_id, cookie)``；拿不到就抛（不许带着空凭据往下跑）。"""
    status, body, set_cookie = _post(base, "/auth/register",
                                     {"username": username, "password": password},
                                     timeout=timeout)
    if status != 201:
        status, body, set_cookie = _post(base, "/auth/login",
                                         {"username": username, "password": password},
                                         timeout=timeout)
    if status not in (200, 201) or not set_cookie:
        raise RuntimeError(f"评测账号登录失败（最后返回 {status}）：{body}")
    user_id = str(body.get("user_id") or username)
    return user_id, set_cookie.split(";", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="给评测账号养长期记忆")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", default="EvalAccount#2026")
    parser.add_argument("--role", default="lawyer")
    parser.add_argument("--session", default="", help="默认 <username>-seed")
    parser.add_argument("--flavor", default="labor", choices=sorted(SEED_FLAVORS),
                        help="这份历史的**性质**（D5 要求换性质不同的历史各跑一轮 A/B）")
    parser.add_argument("--turns", type=int, default=0,
                        help="灌几轮；0 = 该味道的全部轮次（默认）")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    session = args.session or f"{args.username}-seed"
    flavor_desc, flavor_turns = SEED_FLAVORS[args.flavor]
    turns = args.turns if args.turns > 0 else len(flavor_turns)
    user_id, cookie = login(base, args.username, args.password, args.timeout)
    messages = flavor_turns[: max(turns, 1)]
    print(f"评测账号 {user_id}（会话 {session}）将灌入 {len(messages)} 轮背景对话")
    print(f"  历史性质（--flavor {args.flavor}）：{flavor_desc}")
    print(f"  该味道共 {len(flavor_turns)} 轮，本次取前 {len(messages)} 轮")

    try:
        before = _longterm_chunks(base)
    except Exception as exc:  # noqa: BLE001
        print(f"!! 读 /health 失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"  灌之前 longterm.chunks = {before}")

    failures = 0
    for index, message in enumerate(messages, 1):
        status, body, _ = _post(base, "/chat", {
            "user_id": user_id, "role_id": args.role, "session_id": session,
            "message": message, "stream": False}, cookie=cookie, timeout=args.timeout)
        answer = str((body or {}).get("answer") or "")
        if status != 200:
            failures += 1
            print(f"  [{index}/{len(messages)}] 失败 {status}：{(body or {}).get('_error')}")
        else:
            print(f"  [{index}/{len(messages)}] ok {len(answer)} 字")

    try:
        after = _longterm_chunks(base)
    except Exception as exc:  # noqa: BLE001
        print(f"!! 读 /health 失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"  灌之后 longterm.chunks = {after}（+{after - before}）")
    if failures:
        print(f"!! 有 {failures} 轮失败，记忆可能没养够", file=sys.stderr)
        return 1
    if after <= before:
        if before > 0:
            # **不是错误**：同一会话同样的内容会走幂等 upsert（message_id 由内容派生），
            # 第二次灌不涨条数是正确行为。只要库里本来就有记忆，这一轮就是要的效果。
            print(f"  [注意] 条数没变（{before} -> {after}）：同会话同内容会幂等 upsert，"
                  f"库里本来就有记忆，A/B 仍然是有效的")
        else:
            print("!! 记忆条数没有增长（滑窗没触发沉淀），A/B 会退化成'两臂都是空'",
                  file=sys.stderr)
            return 1
    print("SEED-DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
