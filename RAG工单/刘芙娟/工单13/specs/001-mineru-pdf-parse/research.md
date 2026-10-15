# Phase 0 Research: MinerU PDF 解析

**Feature**: `specs/001-mineru-pdf-parse` | **Date**: 2026-09-22

本文件记录所有影响实现的技术决策，每条给出**依据**与**不确定性标注**。凡属推测的，一律显式标记为"需实测"。

---

## 决策 1：MinerU 包版本锁定为 3.x

- **Decision**: 依赖清单写 `mineru>=3.2.0,<4.0`，安装后实测回填精确版本号。
- **Rationale**:
  - 本机权重 `MinerU2.5-Pro-2605-1.2B` 发布于 2026-05-21，属 3.x 版本线的模型集合；
  - 3.x 支持通过 `mineru.json` 的 `models-dir` 指向**任意本地权重目录**，这是离线复用的前提；
  - `docs/04 §5` 已明确"禁止使用 4.x，直到其契约稳定"，且 4.x 换了模型集合与 CLI（`mineru-kit`），破坏性变更。
- **Alternatives rejected**: 2.7.6 及更早（不支持该权重）；4.x（契约未稳，违反 `docs/04 §5`）。
- **来源**: [PyPI mineru](https://pypi.org/project/mineru/)、[MinerU 3.4.5 文档](https://github.com/opendatalab/MinerU/tree/mineru-3.4.5-released/docs/en)
- **不确定性**: 精确到补丁号的版本号需安装后实测确认。

---

## 决策 2：默认后端 `pipeline`，`--backend` 保留 `vlm-engine`

- **Decision**: 脚本默认 `pipeline`，命令行可切 `--backend vlm-engine`。
- **Rationale**（本机实测 + 语料实测共同支撑）:

| 维度 | pipeline | vlm-engine | 对本案的影响 |
|---|---|---|---|
| 文字来源 | PDF 文本层直接抽取（`txt` 模式，OCR 不触发） | 页面渲染为图片后由大模型重新识别 | 语料实测"文本可无损提取"（`docs/04 §3`），VLM 相当于**对无损文本再做一次有损识别** |
| 含 NVIDIA CPU 加速支持 | 官方支持 | **官方不支持**（Windows + 纯 CPU） | 本机实测无 NVIDIA 显卡，仅有 Intel Iris Xe 核显 |
| 噪声类型粒度 | `header`/`footer`/`page_number`/`aside_text`/`page_footnote` 五类具体类型 | 类型集合不同（需实测） | `docs/04 §6` 要求 `drop_rule` 精确到命中规则，否则误杀无法定位 |
| 可复现性 | 确定性模型串联 | 生成式解码 | constitution N4 要求拒答可复现 |
| 与 `docs/04 §12` 一致性 | 一致 | 需回写文档 | pipeline 免去文档变更 |
| 额外下载 | **无**（两套权重均已在盘） | 无 | 该项对两者等价 |

- **Alternatives rejected**: 默认 `vlm-engine`——用户曾据此意向，但在"本机无 CUDA + 官方不支持 + 语料为原生数字版"三项实测事实成立后被推翻。
- **来源**: [MinerU 支持矩阵讨论](https://github.com/opendatalab/MinerU/discussions/4574)、本机 `Get-CimInstance Win32_VideoController` 实测
- **不确定性**: ⚠️ `vlm-engine` 在本机能否跑通未验证；本特性不做实测（用户要求不执行脚本）。

---

## 决策 3：用 `MINERU_TOOLS_CONFIG_JSON` 环境变量覆盖配置，不修改全局 `mineru.json`

- **Decision**: 脚本在仓库内生成 `.mineru/mineru.json`，并把 `MINERU_TOOLS_CONFIG_JSON` 指向它的绝对路径。
- **Rationale**:
  - 源码为 `CONFIG_FILE_NAME = os.getenv('MINERU_TOOLS_CONFIG_JSON', 'mineru.json')`，随后与用户主目录做 `os.path.join`；传入绝对路径时 `os.path.join` 在 Windows 上会直接返回该绝对路径，故可绕过 `C:\Users\Lenovo\mineru.json`；
  - 用户明确要求"不碰全局配置"；
  - 项目内配置文件可随仓库版本化，符合 constitution 的"参数显式记录"要求。
- **Alternatives rejected**: 直接改写 `C:\Users\Lenovo\mineru.json`（用户明确拒绝）；不写配置文件只靠环境变量（MinerU 无对应环境变量可单独指定 `models-dir`）。
- **来源**: [config_reader.py 源码镜像](https://gitcode.com/OpenDataLab/MinerU/blob/eed479eb56bba93ee99c1a8c255d509bd2f837e5/mineru/utils/config_reader.py)、[模型指定讨论 #4331](https://github.com/opendatalab/MinerU/discussions/4331)、[自定义权重目录讨论 #2960](https://github.com/opendatalab/MinerU/discussions/2960)
- **不确定性**: `os.path.join(home, abs_path)` 的绝对路径短路行为在 Windows 上成立，但需实测确认 3.x 未对返回值另作处理。

---

## 决策 4：pipeline 权重指向 C: 缓存（完整），而非 E: 副本（残缺）

- **Decision**: `models-dir.pipeline` 指向 `C:\Users\Lenovo\.cache\modelscope\models\OpenDataLab--PDF-Extract-Kit-1.0\snapshots\master`。
- **Rationale**（本机实测比对）:

| 子目录 | C: 缓存（2.1 G） | E: 副本（1.1 G） |
|---|---|---|
| `Layout/PP-DocLayoutV2` | ✅ | ✅ |
| `Layout/YOLO` | ✅ | ❌ |
| `MFD/YOLO` | ✅ | ❌ |
| `MFR/unimernet_hf_small_2503` | ✅ | ✅ |
| `OCR/paddleocr_torch` | ✅ | ✅ |
| `OriCls/paddle_orientation_classification` | ✅ | ❌ |
| `ReadingOrder/layout_reader` | ✅ | ❌ |
| `TabCls/paddle_table_cls` | ✅ | ✅ |
| `TabRec/SlanetPlus`、`TabRec/UnetStructure` | ✅ | ✅ |

  E: 副本缺 4 个叶子目录，且现有全局 `mineru.json` 正指向该残缺副本。
- **Alternatives rejected**: 指向 E:（残缺，可能导致运行期联网补齐，违反 `docs/04 §11`）。
- **不确定性**: ⚠️ MinerU 3.x 实际需要哪些子模型未经验证。C: 是超集，取超集是安全侧选择。

---

## 决策 5：强制 `model-source: local`，禁止联网兜底

- **Decision**: 项目内配置写 `"model-source": "local"`，并设 `MINERU_MODEL_SOURCE=local`。
- **Rationale**: 现有全局配置的 `model-source` 是 `modelscope`，缺模型时会**联网下载**；`docs/04 §11` 明令"MinerU 权重路径不存在 → 报错退出，MUST NOT 回退到联网下载"。
- **来源**: [离线/断网讨论 #3797](https://github.com/opendatalab/MinerU/discussions/3797)

---

## 决策 6：设 `MINERU_DEVICE_MODE=cpu`

- **Decision**: 显式设为 `cpu`。
- **Rationale**: `-d/--device` 参数自 3.0.0 起已删除，改用该环境变量；本机无 NVIDIA 显卡，显式设 `cpu` 避免 MinerU 探测 CUDA 时的告警或异常路径。
- **来源**: [3.0.0 cli_tools.md](https://github.com/opendatalab/MinerU/blob/mineru-3.0.0-released/docs/en/usage/cli_tools.md)

---

## 决策 7：以子进程调用 `mineru` 命令行，而非 Python API

- **Decision**: 通过固定解释器 `rag/python.exe` 对应的 `Scripts/mineru.exe` 以子进程方式调用。
- **Rationale**:
  - 3.x 的 Python 内部 API（如 `mineru.cli.common.do_parse`）**无官方文档**，签名随版本变动；
  - CLI（`-p`/`-o`/`-b`）是官方文档化的稳定契约；
  - 子进程隔离：MinerU 崩溃/内存溢出不影响主进程，且退出码可精确映射为脚本的退出码。
- **Alternatives rejected**: 直接 `import mineru.cli.common`（未文档化）；调 HTTP API（需额外起服务，复杂度超出 300 行预算）。
- **来源**: [3.4.5 cli_tools.md](https://github.com/opendatalab/MinerU/blob/mineru-3.4.5-released/docs/en/usage/cli_tools.md)
- **不确定性**: ⚠️ 3.4.5 的 `mineru` CLI 会临时启动本地 `mineru-api` 编排；该行为对启动耗时的影响与稳定性未经验证。

---

## 决策 8：`doc_id` = PDF 内容 sha256 前 12 位

- **Decision**: 沿用 `docs/04 §10.1` 的定义，但来源改为直接计算文件内容，不读 manifest。
- **Rationale**: 用户要求"后续继续上传 PDF"，不引入 manifest 准入闸门；内容哈希保证"同名不同内容"不混入，且天然支持幂等跳过。
- **偏离标注**: 本决策**有意偏离** `docs/04 §4`（S1 语料准入），须人工记入该文档。详见 `spec.md` 的 Assumptions 与 checklist 的"有意偏离"节。

---

## 决策 9：产物结构沿用 `docs/04 §5`

- **Decision**: 产物落 `data/parsed/{doc_id}/`，其中 `content_list.json` 为下游唯一消费对象，`middle.json` 供调试，Markdown 仅作人工抽查。
- **Rationale**: `docs/04 §5` 明确"MinerU 的 Markdown 输出不携带页码信息"，页码是本项目引用功能的物理前提。
- **不确定性**: ⚠️ MinerU 各后端的实际输出文件名与层级需实测确认（`docs/04 §15` 待验证项 P1）。

---

## 遗留待实测项（对应 `docs/04 §15`）

| 编号 | 待验证内容 | 验证方式 |
|---|---|---|
| R1 | `mineru` 精确版本号 | 安装后 `python -c "import mineru; print(mineru.__version__)"` |
| R2 | `content_list.json` 的 `type` 取值集合（pipeline 后端实测） | 脚本内置的类型分布报告 |
| R3 | `vlm-engine` 在本机（Windows + 纯 CPU）能否跑通 | 显式 `--backend vlm-engine` 试跑 |
| R4 | `MINERU_TOOLS_CONFIG_JSON` 绝对路径覆盖是否生效 | 跑一次后检查是否仍读全局配置 |
| R5 | C: 缓存中的 pipeline 模型是否足够（是否需要 MFD/OriCls/ReadingOrder） | 首次解析是否报缺模型 |
