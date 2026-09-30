"""document_versions.version_status 的取值词表（唯一定义处）。

本模块是 document_versions.version_status 取值的唯一定义处。
任何模块要用到审核状态字符串，都必须从这里导入；app/ 下不得再出现
"pending_review" / "approved" / "rejected" 裸字面量（含 SQL 条件、接口默认值、
测试夹具之外的业务代码）。

为什么下沉到数据层（app/db/）而不是留在 app/review/：
状态值是**数据库列的取值域**，它的归属地是实体定义所在层。原先放在 review 层时，
数据层（db/import_service、db/vector_index_service）与检索层（retrieval/keyword_search、
retrieval/vector_search）为了过滤"已发布版本"都必须反向引用 review 层，
形成 db→review、retrieval→review 的反向依赖。下沉后依赖方向变为
review/retrieval → db（允许方向），db 内部自引用，不存在任何反向依赖
（由 tests/test_review_status.py 的分层方向测试守住）。

本模块**只放常量**：不含函数、类与任何 import，因此可被任意层安全引用，
既不会形成循环依赖，也不会把上层逻辑带进下层。

值域对应数据库列 document_versions.version_status；
状态机（6.3 / 6.4）见 docs/接口文档.md。
"""

# 待审核：新导入的版本一律进入该状态，等待管理员审核（内容不进 Milvus）
PENDING_REVIEW = "pending_review"

# 已批准：审核通过且向量索引校验成功的版本才会进入生效知识库
APPROVED = "approved"

# 已驳回：该版本向量已回收（MySQL 正文保留，用于留痕与回看），不参与检索
REJECTED = "rejected"

# 合法状态全集：由上面三个常量派生（常量而非逻辑，无副作用）。
# 放在同一模块，才能保证"值域只有一个定义处"；6.3 列表接口的 status 入参
# 校验与错误提示拼装都取它，不再各自拼 tuple。
REVIEW_STATUSES = (PENDING_REVIEW, APPROVED, REJECTED)
