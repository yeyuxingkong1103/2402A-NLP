# 6 部 status 回填 + ① 公布日期修正 —— 执行报告

- 执行时间：2026-09-21 22:02（apply commit 完成）
- 脚本：`scripts/migrations/backfill_status_6laws.py`（新增，走 `scripts/_env.py` 零口令连接，词表常量取自 `app/db/law_status.py`）
- 证据链：`reports/metadata_verification_6laws.md`（2026-09-21 核验，用户已确认证据链成立）
- 词表映射（用户裁决②）：flk「有效」/ 人社部「是否有效：有效」→ `现行有效`；`revision_note` 保留原始标注文字 + URL + 抓取时间

## 一、逐行执行结果（--apply 真实输出）

```
== 逐行写入 ==
      ver_id=91  status -> 现行有效（rowcount=1）
      ver_id=91  promulgation_date 2025-08-01 -> 2025-07-31（rowcount=1）
      ver_id=92  status -> 现行有效（rowcount=1）
      ver_id=95  status -> 现行有效（rowcount=1）
      ver_id=97  status -> 现行有效（rowcount=1）
      ver_id=98  status -> 现行有效（rowcount=1）
      ver_id=99  status -> 现行有效（rowcount=1）

== commit 完成 ==
  迁移后 status 分布：
      现行有效             10 行
      不适用              1 行
```

预检记录：6 行回填前 status 全为 NULL、revision_note 全为空、ver_id=91 prom=2025-08-01，与核验表一致。

## 二、验收（裁决 a–d）

| 项 | 要求 | 实测 | 结果 |
|---|---|---|---|
| a | status 分布 = 现行有效 10 / 不适用 1（原 空 6 / 4 / 1） | `现行有效 10 行 / 不适用 1 行`，零 NULL | ✅ |
| b | ver_id=91 promulgation_date = 2025-07-31 | `2025-07-31`（脚本回显 + 直查复核一致） | ✅ |
| c | check_services 退出码 0 | exit=0，`✅ 全部通过`（`reports/b27b_check_services.txt`） | ✅ |
| d | 全量测试 passed 不减 | **652 passed**（= 清理批基线 652） | ✅ |

## 三、revision_note 写入内容（逐条，直查库复核）

6 行 note 均以「来源：flk 时效性标注＝有效（含公布/施行日期与机关）｜URL｜抓取时间」开头，例（ver_id=91 全文）：

> 来源：flk 时效性标注＝有效（公布2025-07-31 施行2025-09-01 司法解释 最高人民法院）｜https://flk.npc.gov.cn/search?searchContent=劳动争议案件适用法律问题的解释｜抓取 2026-09-21 21:40:24；佐证：最高法公告 法释〔2025〕12号 落款2025年7月31日｜https://www.court.gov.cn/zixun/xiangqing/472691.html；再证：data/labor_law_raw/f54ec9c0cf12aa1e1a0e9fa711ab288ca7400827777dc620524cab5c6065f021.html 载「法释〔2025〕12号」「2025年7月31日」「自2025年9月1日起施行」

ver_id=95（部门规章，flk 不收录）：note 首段为「来源：人社部官网部门规章公开目录 是否有效＝有效（废止时间空；劳部发〔1994〕489号）｜mohrss URL｜抓取 2026-09-21 21:47；佐证：中国政府网国家规章库转载页…」，并注明取发布机关字段的原因。

## 四、① 公布日期修正的证据（三重印证）

1. 最高法官网公告原文（court.gov.cn/zixun/xiangqing/472691.html）：「法释〔2025〕12号」落款「2025年7月31日」；
2. flk 检索结果页标注：公布日期 2025-07-31；
3. **本项目原始 HTML**（`data/labor_law_raw/f54ec9c0cf12aa1e….html`，用户独立复核 + 本机 grep 复核）：三处关键字各命中（`法释〔2025〕12号` ×1、`2025年7月31日` ×1、`自2025年9月1日起施行` ×3）。

原库内 2025-08-01 为发布会日期，非公告落款日，属数据录入口径偏差。

## 五、可逆性

- `--rollback` 依据：`reports/law_status_backfill_6laws_20260921_220230.json`（逐行 old_status / old_promulgation_date / old_revision_note）；回滚时校验当前值仍是本脚本写入值，被其它来源改写过的行自动 SKIP。
- 未执行过回滚（脚本范式与上一枚迁移一致：apply → 可 rollback → 可重放；本轮未做破坏性演练，避免再次写库）。

## 六、经验（按裁决④记档，不改 docs/）

**公布日/施行日这类信息，优先从 `data/labor_law_raw/` 的原始 HTML 里取**——零网络、可复核、与库内数据同源；外部数据库（flk / 人社部 / gov.cn）只用于「时效性标注」这类原始页面本身没有的字段。本次 ① 的日期修正即靠原始 HTML 完成第三重印证，成本近乎为零。

## 七、改动清单

| 文件 | 动作 |
|---|---|
| `scripts/migrations/backfill_status_6laws.py` | 新增（277 行，含 --apply/--rollback） |
| `scripts/migrations/README.md` | 登记（执行顺序表 +1 行、逐个说明 +1 节、「五个」→「六个」） |
| `reports/law_status_backfill_6laws_20260921_220230.json` | 脚本自动生成的回滚记录 |
| `reports/b27b_check_services.txt` | 验收 c 证据 |
| MySQL `law_versions` 6 行 | status ×6、revision_note ×6、promulgation_date ×1 |

未碰 `docs/`；无临时脚本进 `backend/`；Milvus 与 is_current 未动（6 部本就为 True，词表规则不因「现行有效」改变时效过滤结果，无需重建索引）。
