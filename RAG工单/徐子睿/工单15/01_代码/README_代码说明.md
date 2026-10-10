# 工单 15 · 优化代码说明（可直接并入 RAGFlow）

| 文件 | 作用 | 接入点 |
|---|---|---|
| `query_rewrite.py` | 视觉引用识别（图N/第N页/部件N）+ 生成图像增强查询 | 在 `api/db/services/dialog_service.py :: async_chat()` 里，紧跟 `keyword_extraction` 之后调用 `build_enhanced_queries()` |
| `fusion.py` | 多路召回融合：RRF / 加权 | 在 `rag/nlp/search.py :: Dealer.retrieval()` 里，对「文本路」与「图像增强路」两路结果调用 `reciprocal_rank_fusion()` |
| `rerank_config.py` | 检索/重排/视觉引用参数集中配置（环境变量驱动） | 被 patched 检索层读取 |
| `cross_modal_config.yaml` | 上述配置的 YAML 版本（便于运维改参） | 运维/部署配置 |
| `prompt_multimodal.md` | 跨模态 Prompt 模板（含「结合图纸描述」指令） | `rag/prompts/` 对应模板 |

## 接入步骤（patch 草案）

1. **查询理解**（`dialog_service.async_chat`）：
   ```python
   from query_rewrite import build_enhanced_queries
   queries = build_enhanced_queries(questions[-1], figure_desc_lookup=kb_figure_lookup)
   # queries 可能是 [原问题] 或 [原问题, 增强查询]
   ```
2. **多路召回 + RRF**（`rag/nlp/search.py`）：
   ```python
   from fusion import reciprocal_rank_fusion
   ranks_1 = await self.search(req_for(queries[0]), ...)
   if len(queries) > 1:
       ranks_2 = await self.search(req_for(queries[1]), ...)
       fused = reciprocal_rank_fusion([ranks_1.ids, ranks_2.ids], k=cfg.rrf_k,
                                      weights=cfg.route_weights)
   ```
3. **重排**：知识库绑定 `LLMType.RERANK` 模型即可（`dialog_service.get_models` 已支持 `dialog.rerank_id`）。
4. **Prompt**：命中图像块时，用 `prompt_multimodal.md` 的 `{image_contexts}` 注入图纸描述。

> 单元自测：`python query_rewrite.py`、`python fusion.py` 可直接运行（无外部依赖）。
