# RAGFlow 0.23.1：任务与DeepDoc源码分析

依据本工单“研发/ragflow-main”附带源码，不以其他版本教程代替。工单中的paper_id实际对应parser_id。

`api/db/services/task_service.py:360` 的queue_tasks先按文件type分支：PDF读取页数，默认每任务12页；paper默认22页；one/knowledge_graph或非DeepDOC布局、TOC场景用极大page_size保持整文任务。PDF优先进入PDF分支，因此table的“每3000行”只适用于后续Excel分支，不能说PDF table按3000行分任务。

随后从配置、doc_id和页范围计算xxhash摘要，尝试复用旧任务片段，删除过期索引，批量写任务表，再用REDIS_CONN.queue_product写入Redis Stream。消费者分组、消息ACK/重试和进度状态属于任务执行层，不是上传HTTP请求直接完成所有解析。

`rag/svr/task_executor.py:83` FACTORY把paper、table、one分别交给对应chunker，knowledge_graph落到naive，再按配置运行图谱任务。`do_handle_task:949`校验取消、绑定Embedding并探测向量维度、初始化知识库索引；按任务类型处理dataflow/RAPTOR/GraphRAG等分支，普通路径调用chunker、分词和可选关键词/问题增强，批量Embedding、写向量/文本索引、更新文档chunk/token/进度。任务参数包含tenant_id/kb_id/doc_id，不能跨租户共用未知模型状态。

`deepdoc/parser/pdf_parser.py:55` 的RAGFlowPdfParser加载OCR、布局识别、表格结构识别和上下行拼接XGBoost。PDF先渲染及文字定位，OCR补充识别；布局模型区分正文/标题/表/图/页眉等，再按列和阅读顺序合并；表图提取保留裁剪图及位置，正文裁剪/坐标标记交给后续chunker。低分辨率、图线与标号、版面丢弃和正文/图关联均可能影响结果。

本快照内可见PDF、DOCX、Excel、HTML、JSON、Markdown、PPT、TXT解析器，另外有figure、MinerU、Docling、PaddleOCR、TCADP等适配。适配器存在不表示相应外部模型和服务已经安装，是否调用须看parser_config和模型配置。

实际修复：用户显式单页闭区间[7,7]在原queue_tasks中变成s=6,e=6，任务为空；normalize_pdf_page_range修复为[6,7)。默认“全页到极大值”则仍覆盖总页数，不能将该缺陷扩大解释为默认总是丢末页。新增可选工业保真解析保留整页图像，避免图纸无文字时被当空chunk丢弃；OCR在进程内延迟单例化并串行访问。

正式完整服务未启动时，上述源码修复只有独立单元/语法及扩展流水线验证；正式Redis消费、DeepDoc权重、索引和VLM配置需要完整栈复测，不能由本地页面演示替代。
