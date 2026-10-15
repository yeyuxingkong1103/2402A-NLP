# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
# 模块职责：工单步骤 4「定义损失函数」与步骤 5「定义训练参数」并执行微调，微调模型存入本工单目录
# 使用方式：python train_model.py
# 实现说明：本机 sentence-transformers 3.2.1 的 model.fit 依赖未安装的 datasets 包，
#           而安装 datasets 会降级已验收环境中的 fsspec，因此改用等价的 PyTorch 训练循环

import os                                                       # 导入 os 用于目录与文件处理
import json                                                     # 导入 json 用于训练日志落盘
import time                                                     # 导入 time 用于记录训练耗时
from torch.utils.data import DataLoader                         # 导入 DataLoader 用于批式训练
from transformers import get_linear_schedule_with_warmup        # 导入线性预热调度器，对应 warmuplinear
import embedder                                                 # 导入嵌入模型加载模块
import config                                                   # 导入本工单配置模块


def build_examples(triplets):                                   # 定义训练样本构造函数
    """把三元组字典列表转换为 sentence-transformers 的 InputExample 列表。"""
    from sentence_transformers import InputExample              # 延迟导入样本类型
    examples = []                                               # 初始化样本列表
    for t in triplets:                                          # 遍历每个三元组
        examples.append(InputExample(texts=[t["anchor"],        # 锚点：用户问题
                                            t["positive"],      # 正例：答案所在语料块
                                            t["negative"]]))    # 负例：检索排名靠前的干扰块
    print("[信息] 构造训练样本 %d 条" % len(examples))            # 输出样本数量
    return examples                                             # 返回样本列表


def define_loss(model):                                         # 定义损失函数（工单步骤 4）
    """按工单要求定义损失函数：三元组损失为主，叠加套娃损失形成可截断嵌入。"""
    from sentence_transformers import losses as L               # 延迟导入损失函数集合
    base = L.TripletLoss(                                       # 定义三元组损失
        model=model,                                            # 传入待微调模型
        distance_metric=L.TripletDistanceMetric.COSINE,         # 余弦距离：与检索阶段的相似度口径一致
        triplet_margin=config.TRIPLET_MARGIN)                   # 间隔：拉近正例、推远负例的目标下界
    print("[信息] 损失函数：%s（距离度量 %s，间隔 %.2f）"          # 输出主损失配置
          % (config.LOSS_TYPE, config.DISTANCE_METRIC, config.TRIPLET_MARGIN))  # 填入参数
    if not config.USE_MATRYOSHKA:                               # 未启用套娃损失时
        return base                                             # 直接返回三元组损失
    try:                                                        # 尝试叠加套娃损失
        wrapped = L.MatryoshkaLoss(model, base,                 # 套娃损失包裹基础损失
                                   matryoshka_dims=config.MATRYOSHKA_DIMS)  # 指定截断维度层级
        print("[信息] 叠加套娃损失 MatryoshkaLoss，维度 %s"       # 输出套娃损失配置
              % config.MATRYOSHKA_DIMS)                         # 填入维度列表
        return wrapped                                          # 返回包裹后的损失
    except Exception as e:                                      # 环境不支持时容错
        print("[警告] 套娃损失不可用（%r），退回三元组损失" % e)    # 输出告警
        return base                                             # 退回主损失


def define_training_params():                                   # 定义训练参数（工单步骤 5）
    """集中定义训练超参数，返回参数字典便于日志与报告引用。"""
    params = {"epochs": config.EPOCHS,                          # 训练轮数
              "batch_size": config.BATCH_SIZE,                  # 批大小
              "learning_rate": config.LEARNING_RATE,            # 学习率
              "warmup_ratio": config.WARMUP_RATIO,              # 预热比例
              "max_seq_length": config.MAX_SEQ_LENGTH,          # 最大序列长度
              "optimizer": "AdamW",                             # 优化器
              "scheduler": "warmuplinear",                      # 学习率调度：线性预热
              "use_amp": False}                                 # CPU 训练关闭混合精度
    print("[信息] 训练参数：%s" % json.dumps(params, ensure_ascii=False))  # 输出参数摘要
    return params                                               # 返回参数字典


class LossRecorder(object):                                     # 定义损失记录类
    """记录每个日志点的训练损失与轮次，用于生成训练过程曲线。"""

    def __init__(self):                                         # 构造方法
        self.records = []                                       # 初始化记录列表

    def add(self, epoch, step, loss):                           # 定义记录追加方法
        """追加一条日志点记录，字段与模型原生 fit 日志保持一致。"""
        self.records.append({"epoch": round(float(epoch), 3),   # 当前轮次（支持小数）
                             "step": int(step),                 # 全局步数
                             "loss": round(float(loss), 6)})    # 训练损失
        return self.records[-1]                                 # 返回刚写入的记录


def keep_batch(batch):                                          # 定义直通批处理函数
    """InputExample 不是张量，DataLoader 的默认 collate 会抛 TypeError，这里原样返回整批。"""
    return batch                                                # 直接返回样本列表，交由训练循环自行拆列


def batch_features(model, texts_a, texts_b, texts_c):           # 定义批次特征构造函数
    """把一批三元组文本切成三列并逐列分词，得到损失函数所需的特征列表。"""
    return [model.tokenize(texts_a),                            # 锚点列分词结果
            model.tokenize(texts_b),                            # 正例列分词结果
            model.tokenize(texts_c)]                            # 负例列分词结果


def run_epochs(model, criterion, loader, params, recorder):     # 定义训练轮次执行函数
    """执行全部轮次的手工训练循环，返回每轮平均损失。"""
    import torch                                                # 延迟导入 torch 计算张量与梯度
    torch.manual_seed(config.RANDOM_SEED)                       # 固定随机种子保证可复现
    optimizer = torch.optim.AdamW(model.parameters(),           # 创建 AdamW 优化器
                                  lr=params["learning_rate"])   # 指定学习率
    total_steps = len(loader) * params["epochs"]                # 计算总训练步数
    warmup = int(total_steps * params["warmup_ratio"])          # 按比例换算预热步数
    scheduler = get_linear_schedule_with_warmup(                # 创建线性预热调度器
        optimizer, num_warmup_steps=warmup,                     # 预热步数
        num_training_steps=max(total_steps, 1))                 # 训练总步数（防止为 0）
    model.train()                                               # 切换到训练模式（启用 dropout）
    epoch_losses = []                                           # 每轮平均损失记录
    for epoch in range(params["epochs"]):                       # 遍历每一轮
        running, seen = 0.0, 0                                  # 本轮损失累计与步数
        for step, batch in enumerate(loader):                   # 遍历每个批次
            feats = batch_features(model,                       # 逐列分词构造特征
                                   [e.texts[0] for e in batch],  # 锚点文本列
                                   [e.texts[1] for e in batch],  # 正例文本列
                                   [e.texts[2] for e in batch])  # 负例文本列
            loss = criterion(feats, labels=None)                # 计算损失（TripletLoss 需显式占位标签）
            loss.backward()                                     # 反向传播求梯度
            optimizer.step()                                    # 更新模型参数
            scheduler.step()                                    # 更新学习率
            optimizer.zero_grad()                               # 清空梯度缓存
            running += float(loss)                              # 累计损失
            seen += 1                                           # 累计步数
            recorder.add(epoch + (step + 1) / float(len(loader)),  # 折算当前轮次进度
                         epoch * len(loader) + step + 1,        # 全局步数
                         float(loss))                           # 当前批损失
        avg = running / max(seen, 1)                            # 计算本轮平均损失
        epoch_losses.append(round(avg, 6))                      # 记录本轮平均损失
        print("[信息] 第 %d/%d 轮完成，平均损失 %.6f，学习率 %.2e"  # 输出本轮摘要
              % (epoch + 1, params["epochs"], avg,             # 填入轮次与损失
                 scheduler.get_last_lr()[0]))                   # 填入当前学习率
    model.eval()                                                # 训练结束切回推理模式
    return epoch_losses                                         # 返回每轮平均损失


def measure_triplet_loss(model, examples, margin, limit=64):    # 定义训练损失测量函数
    """按 TripletLoss 定义手工计算平均损失：max(0, d(a,p) - d(a,n) + margin)。"""
    from sentence_transformers import util                      # 延迟导入距离工具
    import torch                                                # 延迟导入 torch 计算张量
    sample = examples[:limit]                                   # 抽样若干样本控制耗时
    a = model.encode([e.texts[0] for e in sample],              # 编码锚点
                     batch_size=64, normalize_embeddings=True,  # 归一化
                     convert_to_tensor=True)                    # 转为张量
    p = model.encode([e.texts[1] for e in sample],              # 编码正例
                     batch_size=64, normalize_embeddings=True,  # 归一化
                     convert_to_tensor=True)                    # 转为张量
    n = model.encode([e.texts[2] for e in sample],              # 编码负例
                     batch_size=64, normalize_embeddings=True,  # 归一化
                     convert_to_tensor=True)                    # 转为张量
    d_ap = 1.0 - util.cos_sim(a, p).diagonal()                  # 锚点与正例的余弦距离
    d_an = 1.0 - util.cos_sim(a, n).diagonal()                  # 锚点与负例的余弦距离
    loss = torch.clamp(d_ap - d_an + margin, min=0.0)           # 三元组损失（含间隔与零截断）
    return float(loss.mean())                                   # 返回平均损失


def train():                                                     # 定义微调主函数
    """执行完整微调流程：加载三元组与模型 → 定义损失与参数 → 训练 → 保存模型与日志。"""
    from sentence_transformers import InputExample              # 校验依赖可导入（缺失则尽早报错）
    del InputExample                                            # 仅用于导入校验，随即释放引用
    if not os.path.isfile(config.TRIPLET_FILE):                 # 三元组数据集缺失
        raise RuntimeError("三元组数据集不存在，请先运行 build_dataset.py")  # 抛出明确错误
    with open(config.TRIPLET_FILE, "r", encoding="utf-8") as f:  # 以 UTF-8 打开三元组文件
        triplets = json.load(f)                                 # 读取三元组列表
    print("[信息] 加载三元组 %d 条" % len(triplets))              # 输出数量
    examples = build_examples(triplets)                         # 构造训练样本
    model = embedder.load_model()                               # 加载基础嵌入模型
    model.max_seq_length = config.MAX_SEQ_LENGTH                # 截断到配置的最大序列长度
    criterion = define_loss(model)                              # 步骤 4：定义损失函数
    params = define_training_params()                           # 步骤 5：定义训练参数
    loader = DataLoader(examples, shuffle=True,                 # 构造批式数据加载器
                        batch_size=config.BATCH_SIZE,           # 指定批大小
                        collate_fn=keep_batch)                  # 使用直通 collate 保留 InputExample
    before = measure_triplet_loss(model, examples, config.TRIPLET_MARGIN)  # 训练前平均损失
    print("[信息] 微调前平均三元组损失：%.6f" % before)            # 输出训练前损失
    recorder = LossRecorder()                                   # 创建损失记录器
    config.ensure_dirs()                                        # 确保输出目录存在
    t0 = time.time()                                            # 记录训练开始时间
    epoch_losses = run_epochs(model, criterion, loader, params, recorder)  # 执行手工训练循环
    cost = time.time() - t0                                     # 计算训练耗时
    after = measure_triplet_loss(model, examples, config.TRIPLET_MARGIN)  # 训练后平均损失
    print("[信息] 微调后平均三元组损失：%.6f（下降 %.6f）"          # 输出训练后损失与降幅
          % (after, before - after))                            # 填入数值
    model.save(config.FINETUNED_MODEL_DIR)                      # 显式保存最终模型（保证产出物完整）
    log = {"triplets": len(triplets),                           # 训练样本数
           "batches_per_epoch": len(loader),                    # 每轮批数
           "params": params,                                    # 训练参数
           "loss_before": round(before, 6),                     # 训练前平均损失
           "loss_after": round(after, 6),                       # 训练后平均损失
           "epoch_losses": epoch_losses,                        # 每轮平均损失
           "train_seconds": round(cost, 2),                     # 训练耗时
           "loss_curve": recorder.records,                      # 逐日志点损失曲线
           "trainer": "manual_torch_loop",                      # 训练器实现方式（替代 model.fit）
           "model_dir": config.FINETUNED_MODEL_DIR}             # 模型输出目录
    with open(config.TRAIN_LOG_FILE, "w", encoding="utf-8") as f:  # 以 UTF-8 打开训练日志
        json.dump(log, f, ensure_ascii=False, indent=1)         # 写入训练日志
    print("[信息] 微调完成，耗时 %.1f 秒，模型已保存至 %s"          # 输出完成信息
          % (cost, config.FINETUNED_MODEL_DIR))                 # 填入耗时与路径
    return log                                                  # 返回训练日志


if __name__ == "__main__":                                      # 仅在直接运行时执行
    train()                                                     # 启动微调
