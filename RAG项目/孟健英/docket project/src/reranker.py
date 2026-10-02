# -*- coding: utf-8 -*-
"""Rerank 重排：bge-reranker-v2 交叉编码器打分（0-1），用于相关性阈值判定。

模型不可用时自动回退到向量余弦分数，保证演示不中断。
"""
import logging  # 日志：模型加载失败等关键事件
import math  # 数学库：手动算 sigmoid 用
from pathlib import Path  # 路径处理：把相对模型路径解析到项目根目录
from typing import List  # 类型注解

from src.config import settings  # 统一配置：重排模型路径、运行设备、阈值、预截断条数

logger = logging.getLogger(__name__)  # 本模块日志器
_model = None  # 模型单例：CrossEncoder 几百 MB，全局只加载一次
_failed = False  # 加载失败标记：失败一次就不再反复尝试（避免每次检索都卡）
_internal_sigmoid = False  # 新版 sentence-transformers 已在 predict 内做过 sigmoid
_ROOT = Path(__file__).resolve().parent.parent  # 项目根目录：本文件在 src/ 下，往上两级


def _model_source() -> str:  # 解析模型来源：本地目录优先，离线也能跑
    """本地目录优先（相对路径按项目根目录解析），否则当作 HF 模型名去下载。"""
    local = Path(settings.reranker_model)  # 配置里的模型路径（可能是本地目录，也可能是 HuggingFace 模型名）
    if not local.is_absolute():  # 相对路径按项目根目录解析
        local = _ROOT / settings.reranker_model  # 拼成绝对路径
    if local.exists():  # 本地有模型文件：离线/内网环境直接加载
        return str(local)  # 用本地模型
    return settings.reranker_model  # 否则当 HF 模型名，联网下载


def _get_model():  # 懒加载重排模型：第一次用到才加载，失败永久回退向量分
    """懒加载 CrossEncoder；失败只提示一次，之后永久回退。"""
    global _model, _failed, _internal_sigmoid  # 声明改模块级变量
    if _model is None and not _failed and settings.reranker_enabled:  # 三条件：没加载过 + 没失败过 + 配置允许
        try:  # 加载可能因缺模型文件/无网络失败，try 住走降级
            from sentence_transformers import CrossEncoder  # 函数内导入：模型库重，不用重排就不占内存；交叉编码器把「查询+文档」拼一起过注意力，输出一个相关度分

            _model = CrossEncoder(_model_source(), device=settings.embedding_device)  # 加载 bge-reranker-v2-m3，跑在配置的设备（CPU/GPU）上
            act = getattr(_model, "default_activation_function", None)  # 探测模型自带激活函数
            _internal_sigmoid = act is not None and act.__class__.__name__ == "Sigmoid"  # 内部已做 sigmoid 的话，归一化时不再重复套
        except Exception as exc:  # 任何加载异常都吞掉转回退
            _failed = True  # 标记失败，之后直接用向量分
            logger.warning("重排模型加载失败，已回退向量分数：%s", exc)  # 降级只提示一次，不刷屏
    return _model  # 返回模型或 None（None 即走向量分回退）


def is_ready() -> bool:  # 重排是否可用（注意：会真实触发懒加载）
    """重排模型是否可用（不阻塞：若后台预热线程还没跑完，也不会触发加载）。"""
    return _get_model() is not None  # 能拿到模型才算就绪


def model_loaded() -> bool:  # 非阻塞版可用性检查
    """非阻塞版：只看内存里 _model 有没有被赋值，绝不触发加载。
    主流程（ui_sidebar）应该用这个，后台预热线程跑完自然变 True。
    """
    return _model is not None  # 只看变量不触发加载，UI 轮询用它


def threshold() -> float:  # 取当前生效的相关性阈值
    """当前生效的相关性阈值（重排分与向量分两套量纲）。"""
    return settings.relevance_threshold if _model is not None else settings.vec_threshold  # 两套阈值：重排分（sigmoid 后 0-1）与向量分（余弦）量纲不同必须分开；低于阈值不作答是防幻觉的第一道防线


def _normalize(values) -> list:  # 把模型输出统一成 0-1 分
    """统一成 0-1 分：模型内部已做 sigmoid（或输出本身就在 0-1）就不再套一层。"""
    raw = [float(v) for v in values]  # 先转 Python float
    if _internal_sigmoid or all(0.0 <= v <= 1.0 for v in raw):  # 模型已做过 sigmoid，或输出本就在 0-1：直接用
        return raw  # 不重复归一化
    return [1 / (1 + math.exp(-v)) for v in raw]  # 否则手动套 sigmoid，把 logits 压到 0-1


def rerank(query: str, candidates: List[dict], top_k: int) -> List[dict]:  # 精排入口：召回后的第二级排序，逐对打分取 Top-K
    """给候选打 0-1 分并排序截断。

    CrossEncoder 每对 (query, text) 都要过一遍交叉注意力，慢是天性。
    这里先按 RRF/向量分预截断到 rerank_top_n（默认 12）再打分，避免给 47 个都打一遍。
    """
    if not candidates:  # 空候选直接返回，省一次模型调用
        return []  # 空进空出
    model = _get_model()  # 取模型（加载失败则为 None，走回退）

    cap = settings.rerank_top_n  # 预截断条数：交叉编码器逐对过模型太贵，先粗排截断；先预截断：按向量分降序取前 N 个送入 CrossEncoder
    if len(candidates) > cap:  # 候选超过上限才截
        candidates = sorted(candidates, key=lambda c: c.get("vec", 0.0), reverse=True)[:cap]  # 按向量分降序取前 N：先粗排后精排，兼顾效果与速度

    if model is None:  # 模型不可用：回退用向量余弦分当相关度
        scored = [{**c, "score": max(0.0, float(c.get("vec", 0.0)))} for c in candidates]  # 向量分可能为负（内积），截到 0 起步
    else:  # 模型可用：逐对精排
        logits = model.predict([(query, c["text"]) for c in candidates])  # 交叉编码器逐对打分：每个「查询+候选」对一起过注意力，比双塔向量召回更准但更慢
        scored = [{**c, "score": s} for c, s in zip(candidates, _normalize(logits))]  # 归一化后的 0-1 分写回候选
    scored.sort(key=lambda x: x["score"], reverse=True)  # 按精排分降序
    return scored[:top_k]  # 截断取 Top-K 交给主链做阈值过滤
