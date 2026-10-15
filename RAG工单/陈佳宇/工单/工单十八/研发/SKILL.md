# Skill‑DocumentQualityAssessmentSkill 文档质量评估技能SOP
## 输入参数
doc_dir：待评估文档的文件夹路径

## 执行步骤
1. 遍历输入目录下全部文档，统计文件格式分布：pdf、docx、md等后缀数量与占比。
2. 针对PDF文档逐页分析：统计每页文本字符数量；低于字符阈值判定扫描页；计算扫描页占比。
    - 扫描页占比>70%：标记Scan_PDF；
    - 部分扫描部分文字：标记混合，加入待确认清单；
    - 全部可提取文字：普通文字PDF。
3. 统计全部文档字符长度；计算P25/P50/P75/P90/P99分位数，长度区间分布。
4. 重复文档检测：
    - MD5哈希：完全字节相同判定完全重复文档；
    - SimHash：计算文档文本simhash，高相似度文档放入【待确认版本冲突列表】，需要人工复核。
5. 敏感信息检测：检测手机号、邮箱、身份证；银行卡检测默认关闭；命中记录保存原文上下文，输出【待审核敏感信息列表】。
6. 对每一份文档输出分类标签：Clean_Markdown / Scan_PDF / Table_Heavy / Image_Heavy / Parse_Failed。
7. 组装完整结构化质检报告返回，报告包含：全局统计、待确认清单、待审核清单、每个文档标签。

## 配置参数（可修改）
- pdf_scan_char_threshold = 150 单页判定扫描页字符阈值
- pdf_scan_ratio_threshold = 0.70 扫描页占比阈值
- simhash_sim_threshold = 0.85 SimHash相似度阈值
- enable_bankcard_detect = false 银行卡检测开关

## 输出说明
报告里面“待确认”“待审核”条目不能直接自动处理，必须人工复核确认。
