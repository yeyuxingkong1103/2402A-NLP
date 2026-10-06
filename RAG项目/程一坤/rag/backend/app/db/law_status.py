"""law_versions.status 的取值词表（唯一定义处）。

本模块是 law_versions.status 取值的唯一定义处。任何模块要用到法规效力状态字符串，
都必须从这里导入；app/ 下不得再出现这四类取值的裸字面量（含注释里的引号形式，
由 tests/test_law_status.py 的源码扫描守住）。

为什么必须单一来源：这批取值原先存在**两套互不认识的词表**——
- 列注释（app/db/law_models.py）写的是**英文**：effective / amended / repealed / superseded
- 抽取器（app/ingest/law_metadata_extractor.py）产出的是**中文**：现行有效 / 已废止 / 已失效
- 消费点（retrieval/context_builder、db/vector_index_service、retrieval/keyword_search）
  只按中文逐字比较

后果是**静默**的：写成英文的取值不会命中"已失效"判断，
一部事实上已废止的法规仍会被判"现行有效"，检索时不加区分地返回。
2026-09-21 批次 26 就真的往库里写了 5 行英文取值（4 个 effective + 1 个 not_applicable），
已由 scripts/migrations/migrate_law_status_vocabulary.py 迁回中文。

取值语义：
- 现行有效：页面明确标注"现行有效"，或经人工核对确认现行文本仍有效
  （现行文本经历过历次修正是"修订沿革"，不影响它当前有效，沿革写进 revision_note）
- 已废止：有明确的废止声明（被明令废止）
- 已失效：页面标注"已失效"（多为有效期届满而自然失去效力）
- 不适用：案例材料等**非规范性文件**，本身不存在"生效/失效"概念。
  它**不参与时效判断**——既不算现行有效，也不算已失效（典型案例没有"被废止"一说）。

失效判定只认 EXPIRED_STATUSES（已废止 / 已失效）。空值与"不适用"都不判失效：
缺数据不能成为静默隐藏条文的理由（否则大部分待补录状态的法规会在时效检索里凭空消失），
"不适用"更不是"已失效"。

本模块**只放常量**：不含函数、类与任何 import，因此可被任意层安全引用
（db / ingest / retrieval），既不会形成循环依赖，也不会把上层逻辑带进下层。
"""

# 现行有效：页面标注"现行有效"，或人工核对确认现行文本有效（修订沿革记在 revision_note）
EFFECTIVE = "现行有效"

# 已废止：有明确废止声明
REPEALED = "已废止"

# 已失效：页面标注"已失效"
LAPSED = "已失效"

# 不适用：案例材料等非规范性文件；不参与时效判断（既不判有效也不判失效）
NOT_APPLICABLE = "不适用"

# 时效判定认作"已失效"的取值全集：
# 三个消费点（context_builder.resolve_current_status、vector_index_service 的
# is_current、keyword_search._resolve_current）都取它，保证"索引一个口径、展示另一个口径"
# 这类偏差不会再出现——历史上就是因为三处各自抄了一份中文元组，才让英文取值静默漏过。
EXPIRED_STATUSES = (REPEALED, LAPSED)

# 合法取值全集（含"不适用"）。列宽 VARCHAR(32) 容得下最长 4 字取值。
# 常量而非逻辑，放在同一模块才能保证"值域只有一个定义处"。
LAW_STATUSES = (EFFECTIVE, REPEALED, LAPSED, NOT_APPLICABLE)

# 抽取器可直接从页面文字里认领的状态字样（按此顺序扫描页面）。
# "不适用"不在其中：它由文书类型（案例材料）判定，不从页面文字里认领——
# 页面不会写"不适用"，硬扫只会在正文里误命中。
PAGE_STATUS_KEYWORDS = (EFFECTIVE, REPEALED, LAPSED)
