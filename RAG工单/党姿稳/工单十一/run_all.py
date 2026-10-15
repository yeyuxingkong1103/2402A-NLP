# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
# 模块职责：一键串联工单要求的八个步骤，按顺序执行并汇总各阶段耗时
# 使用方式：python run_all.py
# 说明：各阶段均带磁盘缓存，重复运行会跳过已完成的阶段

import time                                                     # 导入 time 用于统计各阶段耗时
import config                                                   # 导入本工单配置模块


def step(n, title, func):                                       # 定义阶段执行函数
    """执行单个阶段，打印标题并统计耗时；异常时打印错误但不中断后续阶段之外的流程。"""
    print("\n" + "=" * 62)                                      # 打印分隔线
    print("步骤 %d：%s" % (n, title))                            # 打印阶段标题
    print("=" * 62)                                             # 打印分隔线
    t0 = time.time()                                            # 记录起始时间
    out = func()                                                # 执行阶段函数
    cost = time.time() - t0                                     # 计算耗时
    print("[完成] 步骤 %d 耗时 %.1f 秒" % (n, cost))              # 打印阶段耗时
    return out, cost                                            # 返回结果与耗时


def main():                                                     # 定义主函数
    """按工单要求的顺序执行全部步骤。"""
    config.ensure_dirs()                                        # 确保目录存在
    import gen_dataset                                          # 步骤 1、2 所在模块
    import build_dataset                                        # 步骤 3 所在模块
    import train_model                                          # 步骤 4、5 所在模块
    import run_eval                                             # 步骤 7、8 所在模块
    costs = {}                                                  # 各阶段耗时记录
    _, costs["步骤1-2 数据集与问答对生成"] = step(               # 执行步骤 1、2
        2, "数据集生成 + 问答对生成", lambda: gen_dataset.generate())  # 生成问答对数据集
    triplets, costs["步骤3 数据集与模型加载"] = step(             # 执行步骤 3
        3, "数据集与模型加载 + 三元组构造", lambda: build_dataset.build())  # 构造三元组
    _, costs["步骤4-5 损失函数与训练"] = step(                    # 执行步骤 4、5
        5, "定义损失函数 + 定义训练参数 + 微调", train_model.train)  # 微调模型
    _, costs["步骤6-8 评估器与前后评估"] = step(                  # 执行步骤 6、7、8
        8, "创建评估器 + 微调前评估 + 微调后评估", run_eval.run)   # 评估并对比
    print("\n" + "=" * 62)                                      # 打印分隔线
    print("全流程完成，各阶段耗时（秒）：")                        # 打印汇总标题
    for k, v in costs.items():                                  # 遍历各阶段耗时
        print("  %-28s %8.1f" % (k, v))                         # 对齐打印
    print("=" * 62)                                             # 打印分隔线


if __name__ == "__main__":                                      # 仅在直接运行时执行
    main()                                                      # 启动全流程
