# RRF 后粗排与精排、MinerU 解析为空修复设计

> **范围**：在现有 RAG 项目中，保留 `dense+sparse → RRF` 复合检索作为召回基础，在其后新增可迭代的粗排与精排链路；同时修复 MinerU “解析为空”问题，避免空结果被静默当作正常成功。

## 1. 目标

1. 在不破坏现有问答与入库链路的前提下，提升检索阶段的召回质量与最终引用质量。
2. 将“粗排”和“精排”设计为可迭代优化模块，后续可以替换模型、调整候选规模和阈值，而不需要重写主流程。
3. 修复 MinerU 在部分文档上的空输出问题，确保解析结果为空时具备可诊断兜底，而不是表面成功、实质无块。

## 2. 设计原则

- **召回优先**：先扩大候选集，再做重排，避免相关内容过早被截断。
- **分层清晰**：召回、粗排、精排分离，职责单一。
- **可配置**：关键阈值和候选规模进入配置，支持后续迭代。
- **可测试**：每一层都能单独写测试，不依赖端到端手工验证。
- **可诊断**：MinerU 空结果必须有明确兜底路径与错误线索。

## 3. 现状

当前检索链路为：`Qdrant dense+sparse 检索 → FusionQuery(RRF) → 过滤低分 → 构造引用 → Ollama 问答`。

当前 MinerU 链路为：`doc_analyze_streaming → middle_json 解析 → 若异常或空结果则回退 pypdf`。

问题有两个：

- RRF 之后没有额外重排层，最终引用直接受 top_k 限制，召回空间偏小。
- MinerU 在某些文件上会返回空块，现有逻辑虽然会回退，但对“为什么为空”缺少更明确的结构化诊断。

## 4. 推荐方案

### 4.1 检索重排链路

```text
question
  → BGE-M3 dense+sparse
  → Qdrant RRF 召回
  → retrieval_candidate_k 候选扩大
  → coarse rerank
  → coarse_top_k 截断
  → fine rerank
  → final_top_k 引用
  → QaService 生成回答
```

#### 模块职责

- **召回层**：`backend/app/vector_store.py` 继续负责 RRF 召回，不引入排序策略细节。
- **粗排层**：新增一个重排器接口，第一阶段只负责从较大的候选集里筛出更相关的前 N 条。
- **精排层**：新增第二个重排器接口，对粗排结果做更强排序，输出最终用于引用的结果。
- **编排层**：`backend/app/routes_chat.py` 负责串联召回、粗排、精排、阈值过滤和问答调用。

#### 配置建议

- `retrieval_candidate_k`: RRF 后的候选规模，建议初始值 20 或 30。
- `coarse_top_k`: 粗排保留数量，建议初始值 8 或 10。
- `final_top_k`: 精排后最终进入引用的数量，建议保持现有 6 左右，后续再按评测调整。
- `min_retrieval_score`: 继续保留，作为最终兜底过滤阈值。

#### 迭代方式

- 第一轮迭代优先提高召回率，候选数可以偏大。
- 第二轮再根据评测结果压缩 `coarse_top_k` 或替换精排模型。
- 所有参数都应支持后续在 `AppSettings` 中单独调节。

### 4.2 MinerU 修复链路

```text
pdf_path
  → doc_analyze_streaming
  → middle_json 提取
  → 解析结果为空？
      → 是：记录空结果，进入 pypdf 兜底
      → 否：返回 MinerUBlock[]
```

#### 修复重点

1. **空结果识别**：`_parse_with_mineru()` 返回空列表时，不把它解释成“正常解析成功”。
2. **结构兼容**：`_parse_middle_json()` 继续兼容 `preproc_blocks`、`segmented_content_blocks`、`blocks`、`para_blocks` 等字段，避免 MinerU 输出结构变化导致全空。
3. **诊断信息**：空结果和异常都要带上下文日志，方便后续定位是模型问题、字段变化还是文档本身问题。
4. **兜底保持可用**：仍然回退 `pypdf`，保证构建链路不中断。

## 5. 数据流

### 5.1 检索数据流

1. 用户提问进入 `/api/chat`。
2. Qdrant 返回 RRF 候选集。
3. 候选集先扩大到 `retrieval_candidate_k`。
4. 粗排器对候选集做第一轮重排，输出 `coarse_top_k`。
5. 精排器对粗排结果做第二轮重排，输出 `final_top_k`。
6. 结果经 `min_retrieval_score` 过滤后构造成 citations。
7. `QaService` 使用 citations 生成最终回答。

### 5.2 MinerU 数据流

1. `MinerUParser.parse_pdf()` 调用 MinerU 流水线。
2. `_parse_middle_json()` 解析 `middle_json`。
3. 如果返回空列表，触发空结果兜底说明并回退 `pypdf`。
4. 无论 MinerU 还是兜底，都产出稳定的 `MinerUBlock[]`，供后续清洗和分块。

## 6. 错误处理

- **检索重排失败**：任一重排器报错时，整个问答请求应返回可诊断错误，不允许静默降级成错误排序的引用。
- **候选为空**：直接走现有 fallback answer。
- **MinerU 空结果**：记录 warning，并明确回退到 `pypdf`。
- **MinerU 解析异常**：保留现有异常兜底逻辑，不中断构建任务。

## 7. 测试方案

### 7.1 检索重排测试

- RRF 召回在扩大候选后能保留更多相关结果。
- 粗排对候选集进行稳定排序，并保留预期相关文档。
- 精排输出最终 `final_top_k`，且 `/api/chat` 仅消费最终结果。
- 参数调整后，测试仍能验证候选规模和最终数量。

### 7.2 MinerU 测试

- 当 MinerU 返回空 `pdf_info` 时，解析结果不应被视为成功产出。
- 当 `middle_json` 结构缺少某些块字段时，解析器仍能尽量提取文本。
- 当 MinerU 异常或空结果时，兜底路径仍能返回 `MinerUBlock[]`。

## 8. 非目标

- 不在本轮引入完整多阶段学习排序训练流程。
- 不在本轮改变 Qdrant 存储结构。
- 不在本轮移除现有 RRF 复合检索。
- 不在本轮替换问答模型或改写 prompt 体系。

## 9. 验收标准

- `/api/chat` 可在 RRF 后走粗排与精排链路。
- 相关配置项可通过 `AppSettings` 调整。
- MinerU 空结果问题可被稳定回退并记录诊断信息。
- 相关测试覆盖新增排序链路与 MinerU 空结果场景。
