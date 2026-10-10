# 检索精确度报告（full_04｜英文）

- 题目数：16
- 运行模式：`full_04`

## 一、总体指标

| 指标 | 数值 |
| --- | --- |
| answer_accuracy | 不适用 |
| hit_rate | 0.5 |
| mrr | 0.4375 |
| precision@3 | 0.2917 |
| precision@5 | 0.2219 |
| precision@10 | 0.2219 |
| recall@3 | 0.3229 |
| recall@5 | 0.3542 |
| recall@10 | 0.3542 |
| ndcg@3 | 0.3278 |
| ndcg@5 | 0.3443 |
| ndcg@10 | 0.3443 |
| latency_mean_ms | 4332.5 |
| latency_p50_ms | 2431.6 |
| latency_p95_ms | 20981.6 |
| latency_p99_ms | 20981.6 |

> 说明：英文模式下问题为英文、标准要点（answer_key）为中文，逐字覆盖率无意义，故不判定答案正确性（answer_accuracy 为 null）；检索指标与耗时不受语言影响，照常统计。
> 口径：检索指标只统计与题目同文档（doc_id）的召回块：两本招股书页码大量重叠（如 p39、p72 两本皆有），不按文档过滤会把异库同页块误判为相关而虚高。
> 口径：英文模式下问题为英文、标准要点（answer_key）为中文，逐字覆盖率无意义，故不判定答案正确性（answer_accuracy 为 null）；检索指标与耗时不受语言影响，照常统计。

## 二、逐题结果

### id 1

**问题**：How many shares are being issued by Wuhan P&S Information Technology Co., Ltd., and what percentage of the post-issuance total share capital does that represent?

**答案**：Wuhan P&S Information Technology Co., Ltd. is issuing no fewer than 18,400,000 shares (1,840.00 万股) of common stock, representing no less than 25% of the company's total share capital after the issuance.

Citation sources: 招股说明书1, page 70, text (text#70#258).

**要点覆盖率**：不适用　**判定**：不适用（本语言模式不判答案正确性）　**用时**：20981.6 ms　**后端**：deepseek　**语言**：en

**answerable（结构化）**：True　**拒答判据来源**：structured　**纯正则判定**：非拒答

**标准要点**：1,670万股、占发行后总股本、25.04%

**出处页码**：[2, 22, 24, 306]　**召回页码**：[23, 1, 15, 70, 22]　**计分页码（同文档）**：[23, 1, 15]

**引用来源**：

| 文档 | 页码 | 类型 | 来源ID | 相似度 |
| --- | --- | --- | --- | --- |
| 招股说明书2 | 23 | text | text#23#75 | 0.402286 |
| 招股说明书2 | 1 | text | text#1#0 | 0.257399 |
| 招股说明书2 | 15 | text | text#15#40 | -0.269684 |
| 招股说明书1 | 70 | text | text#70#258 | -1.236653 |
| 招股说明书1 | 22 | text | text#22#78 | -1.654241 |

### id 2

**问题**：Which projects does Wuhan P&S Information Technology Co., Ltd. plan to invest the raised funds in?

**答案**：Based on the retrieved passages, the only raised-fund investment project explicitly named is:

- 基于云联邦架构的军用视频指挥平台升级及产业化项目 (the military video command platform upgrade and industrialization project based on a cloud-federated architecture)

The passages also mention the following projects by name (without stating they are raised-fund projects in the same list):

- 仓储及物流中心项目 (Warehousing and Logistics Center Project)
- 研发中心项目 (R&D Center Project)
- 电子商务平台项目 (E-commerce Platform Project)

Note: The passages do not provide a single complete enumerated list of all raised-fund investment projects, so additional projects, if any, cannot be determined from the document.

Citation sources: 招股说明书1, page 484, 正文(text); 招股说明书2, page 142, 正文(text).

**要点覆盖率**：不适用　**判定**：不适用（本语言模式不判答案正确性）　**用时**：10685.5 ms　**后端**：deepseek　**语言**：en

**answerable（结构化）**：True　**拒答判据来源**：structured　**纯正则判定**：非拒答

**标准要点**：仓储及物流中心、研发中心、电子商务平台、扩充产品种类和数量、其他与主营业务相关的营运资金

**出处页码**：[22, 30, 306]　**召回页码**：[1, 15, 23, 484, 142]　**计分页码（同文档）**：[1, 15, 23, 142]

**引用来源**：

| 文档 | 页码 | 类型 | 来源ID | 相似度 |
| --- | --- | --- | --- | --- |
| 招股说明书2 | 1 | text | text#1#0 | 1.354542 |
| 招股说明书2 | 15 | text | text#15#40 | -0.993176 |
| 招股说明书2 | 23 | text | text#23#75 | -1.853883 |
| 招股说明书1 | 484 | text | text#484#2130 | -2.366246 |
| 招股说明书2 | 142 | text | text#142#675 | -2.722937 |

### id 3

**问题**：Who are the related parties with a controlling relationship to Wuhan P&S Information Technology Co., Ltd., and what are their shareholding percentages and relationship to the company?

**答案**：The related party with a controlling relationship is:

- 赵马克 (Zhao Mark) — 42.35% — 公司控股股东 (controlling shareholder of the company)

Citation: 招股说明书2, page 157, text (正文).

**要点覆盖率**：不适用　**判定**：不适用（本语言模式不判答案正确性）　**用时**：6610.9 ms　**后端**：deepseek　**语言**：en

**answerable（结构化）**：True　**拒答判据来源**：structured　**纯正则判定**：非拒答

**标准要点**：赵马克、42.35%、控股股东、实际控制人

**出处页码**：[21, 33, 157]　**召回页码**：[157, 1, 23, 15, 262]　**计分页码（同文档）**：[157, 1, 23, 15]

**引用来源**：

| 文档 | 页码 | 类型 | 来源ID | 相似度 |
| --- | --- | --- | --- | --- |
| 招股说明书2 | 157 | text | text#157#753 | 2.81674 |
| 招股说明书2 | 1 | text | text#1#0 | 1.90093 |
| 招股说明书2 | 23 | text | text#23#75 | 0.85815 |
| 招股说明书2 | 15 | text | text#15#40 | -0.595098 |
| 招股说明书1 | 262 | text | text#262#1189 | -0.945863 |

### id 4

**问题**：Which related-party enterprises do NOT have a controlling relationship with Wuhan P&S Information Technology Co., Ltd.?

**答案**：佰力电子 (Baili Electronics) — originally controlled by the same actual controller, transferred out in 2008
盈硅电子 (Yinggui Electronics) — originally controlled by the same actual controller, transferred out in 2008
力源科技 (Liyuan Technology) — ceased operating activities since 2001 and its business license was revoked by the Wuhan Administration for Industry and Commerce on April 29, 2005

Citation: 招股说明书2, page 158, text (text#158#754)

**要点覆盖率**：不适用　**判定**：不适用（本语言模式不判答案正确性）　**用时**：3155.8 ms　**后端**：deepseek　**语言**：en

**answerable（结构化）**：True　**拒答判据来源**：structured　**纯正则判定**：非拒答

**标准要点**：融冰投资、听音投资、联众聚源、武汉博润、上海博润、力源贸易、普芯达

**出处页码**：[157]　**召回页码**：[1, 23, 15, 158, 73]　**计分页码（同文档）**：[1, 23, 15, 158]

**引用来源**：

| 文档 | 页码 | 类型 | 来源ID | 相似度 |
| --- | --- | --- | --- | --- |
| 招股说明书2 | 1 | text | text#1#0 | 2.228575 |
| 招股说明书2 | 23 | text | text#23#75 | 0.391323 |
| 招股说明书2 | 15 | text | text#15#40 | 0.367633 |
| 招股说明书2 | 158 | text | text#158#754 | -2.600075 |
| 招股说明书1 | 73 | text | text#73#272 | -5.517171 |

### id 5（严格口径）

**问题**：In the organizational chart of Wuhan P&S Information Technology Co., Ltd., how many departments make up the Sales Department, and how many sales offices make up the Key Account Sales Department?

**答案**：Cannot be determined from the document.

Citation sources: 招股说明书2, page 40, text (text#40#135); 招股说明书1, page 236, text (text#236#1047)

**要点覆盖率**：不适用　**判定**：🚫 拒答（不计正确）　**用时**：1984.8 ms　**后端**：deepseek　**语言**：en

**answerable（结构化）**：False　**拒答判据来源**：structured　**纯正则判定**：非拒答

**标准要点**：渠道销售部、电话及网络销售部、大客户销售部、国际贸易部、北京销售处、深圳销售处、广州销售处、成都销售处、珠海销售处、武汉销售处

**出处页码**：[39]　**召回页码**：[15, 23, 1, 40, 236]　**计分页码（同文档）**：[15, 23, 1, 40]

**引用来源**：

| 文档 | 页码 | 类型 | 来源ID | 相似度 |
| --- | --- | --- | --- | --- |
| 招股说明书2 | 15 | text | text#15#40 | 0.045561 |
| 招股说明书2 | 23 | text | text#23#75 | -0.131371 |
| 招股说明书2 | 1 | text | text#1#0 | -0.370411 |
| 招股说明书2 | 40 | text | text#40#135 | -3.528509 |
| 招股说明书1 | 236 | text | text#236#1047 | -3.740345 |

### id 6

**问题**：According to the 2008 China IC market application structure and growth chart in the prospectus of Wuhan P&S Information Technology Co., Ltd., which industry had the fastest growth rate and which had negative growth?

**答案**：The industry with the fastest growth rate was 汽车 (Automotive) at 14.0%, and the industry with negative growth was IC 卡 (IC Card) at -2.0%.

Citation sources: page 310, image (image#310#fig88); page 72, image (image#72#fig9).

**要点覆盖率**：不适用　**判定**：不适用（本语言模式不判答案正确性）　**用时**：2244.5 ms　**后端**：deepseek　**语言**：en

**answerable（结构化）**：True　**拒答判据来源**：structured　**纯正则判定**：非拒答

**标准要点**：汽车、14.0%、IC卡、-2.0%、负增长

**出处页码**：[72, 310]　**召回页码**：[310, 83, 72, 83, 17]　**计分页码（同文档）**：[310, 83, 72, 83, 17]

**引用来源**：

| 文档 | 页码 | 类型 | 来源ID | 相似度 |
| --- | --- | --- | --- | --- |
| 招股说明书2 | 310 | image | image#310#fig88 | 3.36272 |
| 招股说明书2 | 83 | text | text#83#372 | 3.216279 |
| 招股说明书2 | 72 | image | image#72#fig9 | 3.153791 |
| 招股说明书2 | 83 | image | image#83#fig15 | 3.107514 |
| 招股说明书2 | 17 | text | text#17#48 | 2.521353 |

### id 260

**问题**：During the reporting period, what were the revenues of Wuhan Xingtu Xinke Electronics Co., Ltd. from the military field in each period?

**答案**：During the reporting period, the company's main business revenue from the national defense (military) field in each period was as follows:

- 2019 1-6月: 4,627.14 (94.34%)
- 2018年度: 18,780.67 (94.84%)
- 2017年度: 14,414.16 (97.31%)
- 2016年度: 6,464.51 (82.10%)

Citation sources: 招股说明书1, page 185 (text); 招股说明书1, page 4 (text).

**要点覆盖率**：不适用　**判定**：不适用（本语言模式不判答案正确性）　**用时**：2580.0 ms　**后端**：deepseek　**语言**：en

**answerable（结构化）**：True　**拒答判据来源**：structured　**纯正则判定**：非拒答

**标准要点**：军用领域、6,464.51、14,414.16、18,780.67、4,627.14

**出处页码**：[129, 130]　**召回页码**：[331, 185, 378, 344, 4]　**计分页码（同文档）**：[331, 185, 378, 344, 4]

**引用来源**：

| 文档 | 页码 | 类型 | 来源ID | 相似度 |
| --- | --- | --- | --- | --- |
| 招股说明书1 | 331 | text | text#331#1549 | 1.313555 |
| 招股说明书1 | 185 | text | text#185#883 | 1.103108 |
| 招股说明书1 | 378 | text | text#378#1681 | -0.117775 |
| 招股说明书1 | 344 | text | text#344#1597 | -0.653014 |
| 招股说明书1 | 4 | text | text#4#11 | -0.67033 |

### id 95

**问题**：Which technical standard did Wuhan Xingtu Xinke Electronics Co., Ltd. participate in formulating?

**答案**：The technical standard that Wuhan Xingtu Xinke Electronics Co., Ltd. participated in formulating is a technical standard (技术标准) led by the Academy of Military Sciences (军事科学院), jointly formulated with the Institute of Data Communication Science and Technology (数据通信科学技术研究所) and Peking University (北京大学), and used mainly to guide the development of equipment related to video command systems and system construction. The company participated in the formulation of the technical specification and experimental verification work, along with other domestic mainstream video equipment manufacturers such as 飞讯数码, 维盛网域, 科达, 数码视讯, 海康威视, 浙江大华, 深圳邦彦, and 奥维通信.

Citation sources: page 159 (text), page 160 (text)

**要点覆盖率**：不适用　**判定**：不适用（本语言模式不判答案正确性）　**用时**：2383.5 ms　**后端**：deepseek　**语言**：en

**answerable（结构化）**：True　**拒答判据来源**：structured　**纯正则判定**：非拒答

**标准要点**：视频指挥系统、技术标准、视频技术规范

**出处页码**：[26, 27]　**召回页码**：[1, 159, 52, 160, 1]　**计分页码（同文档）**：[1, 159, 52, 160, 1]

**引用来源**：

| 文档 | 页码 | 类型 | 来源ID | 相似度 |
| --- | --- | --- | --- | --- |
| 招股说明书1 | 1 | text | text#1#0 | -0.16355 |
| 招股说明书1 | 159 | text | text#159#750 | -1.379231 |
| 招股说明书1 | 52 | text | text#52#190 | -2.157877 |
| 招股说明书1 | 160 | text | text#160#752 | -2.422118 |
| 招股说明书1 | 1 | image | image#1#fig0 | -2.553711 |

### id 33

**问题**：During the reporting period, what proportion of main business revenue did military-field revenue represent for Wuhan Xingtu Xinke Electronics Co., Ltd.?

**答案**：During the reporting period, the proportion of main business revenue represented by military-field (defense) revenue was as follows:

- 2016: 82.10%
- 2017: 97.31%
- 2018: 94.84%
- January–June 2019: 94.34%

Citation sources: 招股说明书1, page 185 (text); page 129 (text); page 4 (text).

**要点覆盖率**：不适用　**判定**：不适用（本语言模式不判答案正确性）　**用时**：1921.3 ms　**后端**：deepseek　**语言**：en

**answerable（结构化）**：True　**拒答判据来源**：structured　**纯正则判定**：非拒答

**标准要点**：82.10%、97.31%、94.84%、94.34%

**出处页码**：[129, 130]　**召回页码**：[185, 129, 4, 344, 331]　**计分页码（同文档）**：[185, 129, 4, 344, 331]

**引用来源**：

| 文档 | 页码 | 类型 | 来源ID | 相似度 |
| --- | --- | --- | --- | --- |
| 招股说明书1 | 185 | text | text#185#883 | 2.061717 |
| 招股说明书1 | 129 | text | text#129#580 | 1.541059 |
| 招股说明书1 | 4 | text | text#4#11 | 1.351861 |
| 招股说明书1 | 344 | text | text#344#1597 | 0.86643 |
| 招股说明书1 | 331 | text | text#331#1549 | 0.82094 |

### id 34

**问题**：According to the prospectus of Wuhan Xingtu Xinke Electronics Co., Ltd., which enterprises are involved in the upstream of the electronic information industry?

**答案**：Enterprises involved in the upstream of the electronic information industry include:
- Electronic component manufacturing enterprises related to information systems
- Metal shell (chassis, cabinet) manufacturing enterprises

Citation: Page 152, text (text#152#703).

**要点覆盖率**：不适用　**判定**：不适用（本语言模式不判答案正确性）　**用时**：1670.1 ms　**后端**：deepseek　**语言**：en

**answerable（结构化）**：True　**拒答判据来源**：structured　**纯正则判定**：非拒答

**标准要点**：电子元器件制造企业、机箱、机柜、金属壳体制造企业

**出处页码**：[152, 153]　**召回页码**：[1, 152, 52, 22, 152]　**计分页码（同文档）**：[1, 152, 52, 22, 152]

**引用来源**：

| 文档 | 页码 | 类型 | 来源ID | 相似度 |
| --- | --- | --- | --- | --- |
| 招股说明书1 | 1 | text | text#1#0 | 2.268817 |
| 招股说明书1 | 152 | text | text#152#703 | 1.581692 |
| 招股说明书1 | 52 | text | text#52#190 | 0.341195 |
| 招股说明书1 | 22 | text | text#22#78 | -0.226267 |
| 招股说明书1 | 152 | text | text#152#704 | -0.368372 |

### id 957

**问题**：In which field has Wuhan Xingtu Xinke Electronics Co., Ltd. become an important supplier?

**答案**：Wuhan Xingtu Xinke Electronics Co., Ltd. has become an important supplier in the field of military video command (军队视频指挥领域).

Citation: 招股说明书1, page 95, text (正文).

**要点覆盖率**：不适用　**判定**：不适用（本语言模式不判答案正确性）　**用时**：2249.8 ms　**后端**：deepseek　**语言**：en

**answerable（结构化）**：True　**拒答判据来源**：structured　**纯正则判定**：非拒答

**标准要点**：视频指挥、重要供应商

**出处页码**：[26, 27, 95]　**召回页码**：[95, 1, 52, 1, 22]　**计分页码（同文档）**：[95, 1, 52, 1, 22]

**引用来源**：

| 文档 | 页码 | 类型 | 来源ID | 相似度 |
| --- | --- | --- | --- | --- |
| 招股说明书1 | 95 | text | text#95#393 | 3.560752 |
| 招股说明书1 | 1 | text | text#1#0 | 1.187826 |
| 招股说明书1 | 52 | text | text#52#190 | -0.340603 |
| 招股说明书1 | 1 | image | image#1#fig0 | -1.830816 |
| 招股说明书1 | 22 | text | text#22#78 | -1.840187 |

### id 793

**问题**：According to the prospectus of Wuhan Xingtu Xinke Electronics Co., Ltd., what industries mainly constitute the downstream of the electronic information industry?

**答案**：The downstream of the electronic information industry mainly consists of:
- Military (军队)
- Government agencies (政府机关)
- Energy and other industry enterprises (能源等行业企业)
- Various end users (各类终端用户)

Citation sources: 招股说明书1, page 152, text (text#152#703); 招股说明书1, page 153, figure text (figuretext#153#fig38).

**要点覆盖率**：不适用　**判定**：不适用（本语言模式不判答案正确性）　**用时**：2479.7 ms　**后端**：deepseek　**语言**：en

**answerable（结构化）**：True　**拒答判据来源**：structured　**纯正则判定**：非拒答

**标准要点**：军队、政府、能源

**出处页码**：[152, 153]　**召回页码**：[152, 152, 1, 153, 52]　**计分页码（同文档）**：[152, 152, 1, 153, 52]

**引用来源**：

| 文档 | 页码 | 类型 | 来源ID | 相似度 |
| --- | --- | --- | --- | --- |
| 招股说明书1 | 152 | text | text#152#703 | 3.636528 |
| 招股说明书1 | 152 | text | text#152#704 | 1.692687 |
| 招股说明书1 | 1 | text | text#1#0 | 1.677122 |
| 招股说明书1 | 153 | text | figuretext#153#fig38 | 1.583815 |
| 招股说明书1 | 52 | text | text#52#190 | 0.512351 |

### id 795

**问题**：Which project that Wuhan Xingtu Xinke Electronics Co., Ltd. participated in won the First Prize of the National Science and Technology Progress Award?

**答案**：Cannot be determined from the document.

Citation sources: 招股说明书1, page 1 (text), page 1 (image), page 22 (text), page 22 (table), page 52 (text).

**要点覆盖率**：不适用　**判定**：🚫 拒答（不计正确）　**用时**：3761.3 ms　**后端**：deepseek　**语言**：en

**answerable（结构化）**：False　**拒答判据来源**：structured　**纯正则判定**：非拒答

**标准要点**：情报、通信网络、一体化工程、C4ISR、国家科技进步一等奖

**出处页码**：[27, 94, 96]　**召回页码**：[1, 52, 22, 22, 1]　**计分页码（同文档）**：[1, 52, 22, 22, 1]

**引用来源**：

| 文档 | 页码 | 类型 | 来源ID | 相似度 |
| --- | --- | --- | --- | --- |
| 招股说明书1 | 1 | text | text#1#0 | 3.16659 |
| 招股说明书1 | 52 | text | text#52#190 | -0.444463 |
| 招股说明书1 | 22 | table | table#22#10 | -1.385862 |
| 招股说明书1 | 22 | text | text#22#78 | -1.455254 |
| 招股说明书1 | 1 | image | image#1#fig0 | -1.683682 |

### id 543

**问题**：What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?

**答案**：The registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd. is 5,520.00万元 (RMB 55.20 million).

Citation sources: 招股说明书1, page 22, text; 招股说明书1, page 22, table; 招股说明书1, page 52, text.

**要点覆盖率**：不适用　**判定**：不适用（本语言模式不判答案正确性）　**用时**：2981.8 ms　**后端**：deepseek　**语言**：en

**answerable（结构化）**：True　**拒答判据来源**：structured　**纯正则判定**：非拒答

**标准要点**：5,520

**出处页码**：[22, 52]　**召回页码**：[52, 22, 15, 22, 1]　**计分页码（同文档）**：[52, 22, 22, 1]

**引用来源**：

| 文档 | 页码 | 类型 | 来源ID | 相似度 |
| --- | --- | --- | --- | --- |
| 招股说明书1 | 52 | text | text#52#190 | 6.622168 |
| 招股说明书1 | 22 | text | text#22#78 | 2.962658 |
| 招股说明书2 | 15 | text | text#15#40 | 2.81363 |
| 招股说明书1 | 22 | table | table#22#10 | 2.084662 |
| 招股说明书1 | 1 | text | text#1#0 | 1.806938 |

### id 531

**问题**：Who is the legal representative of Wuhan Xingtu Xinke Electronics Co., Ltd.?

**答案**：The legal representative of Wuhan Xingtu Xinke Electronics Co., Ltd. is 程家明 (Cheng Jiaming).

Citation sources: 招股说明书1, page 52 (text); 招股说明书1, page 22 (table); 招股说明书1, page 22 (text).

**要点覆盖率**：不适用　**判定**：不适用（本语言模式不判答案正确性）　**用时**：1857.3 ms　**后端**：deepseek　**语言**：en

**answerable（结构化）**：True　**拒答判据来源**：structured　**纯正则判定**：非拒答

**标准要点**：程家明

**出处页码**：[22, 52]　**召回页码**：[52, 22, 22, 15, 1]　**计分页码（同文档）**：[52, 22, 22, 1]

**引用来源**：

| 文档 | 页码 | 类型 | 来源ID | 相似度 |
| --- | --- | --- | --- | --- |
| 招股说明书1 | 52 | text | text#52#190 | 5.826652 |
| 招股说明书1 | 22 | table | table#22#10 | 2.45505 |
| 招股说明书1 | 22 | text | text#22#78 | 2.208061 |
| 招股说明书2 | 15 | text | text#15#40 | 1.341287 |
| 招股说明书1 | 1 | text | text#1#0 | -1.214432 |

### id 207

**问题**：How much of the raised funds does Wuhan Xingtu Xinke Electronics Co., Ltd. plan to use to supplement working capital?

**答案**：Wuhan Xingtu Xinke Electronics Co., Ltd. plans to use RMB 15,000万元 (150 million yuan) of the raised funds to supplement working capital.

Citation: 招股说明书1, page 490, text (正文).

**要点覆盖率**：不适用　**判定**：不适用（本语言模式不判答案正确性）　**用时**：1772.9 ms　**后端**：deepseek　**语言**：en

**answerable（结构化）**：True　**拒答判据来源**：structured　**纯正则判定**：非拒答

**标准要点**：15,000、补充流动资金

**出处页码**：[23, 30, 479]　**召回页码**：[1, 52, 22, 22, 490]　**计分页码（同文档）**：[1, 52, 22, 22, 490]

**引用来源**：

| 文档 | 页码 | 类型 | 来源ID | 相似度 |
| --- | --- | --- | --- | --- |
| 招股说明书1 | 1 | text | text#1#0 | -1.650382 |
| 招股说明书1 | 52 | text | text#52#190 | -1.692439 |
| 招股说明书1 | 22 | text | text#22#78 | -3.133747 |
| 招股说明书1 | 22 | table | table#22#10 | -3.293733 |
| 招股说明书1 | 490 | text | text#490#2158 | -3.306081 |
