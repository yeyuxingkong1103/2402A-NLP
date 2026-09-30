"""General medical image analysis routes."""
import base64
import json
import os

from fastapi import APIRouter, Depends, HTTPException

from .auth import get_current_user
from .dependencies import PIPELINE_AVAILABLE, _llm_supports_images, get_llm, get_pipeline
from .history import _append_message, _resolve_conversation
from .schemas import ImageRequest

router = APIRouter(tags=["vision"])


def _vision_key() -> str:  # 获取视觉模型 API Key
    """视觉模型与 RAG/图检索共用统一模型配置。"""
    try:  # 尝试读配置
        from src import config as _cfg  # 导入配置

        return _cfg.LLM_API_KEY  # 返回大模型 Key
    except Exception:  # 读取失败
        return ""  # 返回空串


def _ask_qwen_vision(image_b64: str, filename: str, question: str = "") -> str | None:  # 调用项目 LLM 看图
    ext = os.path.splitext(filename)[1].lower().lstrip(".")  # 取扩展名并小写
    mime = {  # 扩展名 → MIME 类型映射
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
        "bmp": "image/bmp",
    }.get(ext, "image/jpeg")  # 取不到默认 jpeg
    prompt = (  # 组装给模型的提示词
        (f"用户还问了：{question}\n" if question else "")
        + "请观察这张照片，描述你看到的情况，并给出谨慎的健康建议。"
        "注意：不要下确定诊断，强调仅供参考，建议及时就医。"
    )
    try:  # 尝试用项目 LLM
        if _llm_supports_images():  # 支持图片输入
            return get_llm().chat(  # 调用统一模型层
                system="你是专业的医疗健康咨询助手。",
                user=prompt,
                images=[{"media_type": mime, "data": image_b64}],
            )
    except Exception:  # 调用失败
        pass  # 忽略，走直连回退
    # 项目 LLM 不支持图片时，回退到原有直连方式
    return _ask_qwen_vision_direct(image_b64, filename, question)  # 直连千问


def _ask_qwen_vision_direct(
    image_b64: str, filename: str, question: str = ""
) -> str | None:
    """直连千问视觉模型（项目 LLM 不支持图片时的回退）。"""
    api_key = _vision_key()  # 取 Key
    if not api_key:  # 没有 Key
        return None  # 无法调用
    try:  # 尝试直连
        from openai import OpenAI  # 导入 OpenAI 客户端

        client = OpenAI(  # 创建客户端
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
        )
        ext = os.path.splitext(filename)[1].lower().lstrip(".")  # 取扩展名
        mime = {  # 扩展名 → MIME
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
            "gif": "image/gif",
            "bmp": "image/bmp",
        }.get(ext, "image/jpeg")  # 默认 jpeg
        completion = client.chat.completions.create(  # 调用聊天补全接口
            model="qwen-vl-plus",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{image_b64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                (f"用户还问了：{question}\n" if question else "")
                                + "请观察这张照片，描述你看到的情况，并给出谨慎的健康建议。"
                                "注意：不要下确定诊断，强调仅供参考，建议及时就医。"
                            ),
                        },
                    ],
                }
            ],
            temperature=0.3,
        )
        return completion.choices[0].message.content  # 返回模型回答文本
    except Exception:  # 调用异常
        return None  # 返回空


@router.post("/api/analyze-image")  # 图片识别接口
def analyze_image(req: ImageRequest, _user: dict | None = Depends(get_current_user)):  # 登录校验
    """上传/拍照图片，进行谨慎的看图咨询。"""
    image = req.image.strip()  # 去掉空白
    if not image or len(image) < 32:  # 内容过短
        raise HTTPException(status_code=400, detail="图片内容为空或无效")  # 拒绝
    if len(image) > 12 * 1024 * 1024:  # base64 过大
        raise HTTPException(status_code=400, detail="图片过大，请选择 10MB 以内的图片")  # 拒绝
    try:  # 校验 base64 合法性
        base64.b64decode(image, validate=True)  # 严格解码
    except Exception:  # 解码失败
        raise HTTPException(status_code=400, detail="图片编码无效")  # 拒绝

    filename = req.filename or "photo.jpg"  # 取文件名
    ext = os.path.splitext(filename)[1].lower().lstrip(".")  # 取扩展名
    mime = {  # 扩展名 → MIME
        "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
        "webp": "image/webp", "gif": "image/gif", "bmp": "image/bmp",
    }.get(ext, "image/jpeg")  # 默认 jpeg

    user_id = _user["id"] if _user else None  # 当前用户 ID
    conversation_id = _resolve_conversation(  # 找到/新建历史对话
        user_id,
        req.conversation_id,
        req.question.strip() or "图片识别",
        "image",
    )
    # 把上传的图片存进历史（dataUrl 形式，供打开历史时显示）
    attachment = json.dumps(  # 单张图片附件
        [{"name": filename, "type": "image", "dataUrl": f"data:{mime};base64,{image}"}],
        ensure_ascii=False,
    )
    _append_message(  # 用户问题写入历史（没打字就存空字符串，历史里只显示图片）
        conversation_id,
        "user",
        req.question.strip(),
        attachment,
    )

    try:  # 优先走检索管线（带图多模态）
        if PIPELINE_AVAILABLE:  # 管线可用
            result = get_pipeline().generate_answer(  # 调用管线多模态生成
                req.question.strip(),
                images=[{"media_type": mime, "data": image}],
            )
            answer = result.get("answer", {})  # 取答案对象
            answer_text = answer.get("answer", "") if isinstance(answer, dict) else str(answer)  # 取文本
            if answer_text.strip():  # 有内容
                _append_message(conversation_id, "assistant", answer_text)  # 写入历史
                response = {  # 组装成功响应
                    "ok": True,
                    "source": "基于知识库检索生成",
                    "answer": answer_text,
                    "suggestions": [],
                    "conversation_id": conversation_id,
                }
                return response  # 返回
    except Exception:  # 管线失败
        pass  # 忽略，走视觉模型

    answer = _ask_qwen_vision(image, filename, req.question)  # 调用视觉模型
    if answer:  # 有回答
        _append_message(conversation_id, "assistant", answer)  # 写入历史
        return {  # 返回
            "ok": True,
            "source": "AI 看图分析（仅供参考）",
            "answer": answer,
            "suggestions": [],
            "conversation_id": conversation_id,
        }
    fallback_answer = (  # 兜底文案（服务不可用时）
        "当前 AI 图片识别服务暂时不可用（通常是账户欠费或网络问题），"
        "我还没能真正分析照片内容。\n\n"
        "你可以先用文字描述一下症状或药品外观，我会通过知识图谱帮你查询。\n\n"
        "紧急情况（红肿发热、疼痛明显、影响呼吸等）请直接就医。"
        "本功能为学习演示，不构成医学诊断。"
    )
    _append_message(conversation_id, "assistant", fallback_answer)  # 兜底也写入历史
    return {  # 返回兜底
        "ok": True,
        "source": "图片识别服务暂不可用",
        "answer": fallback_answer,
        "suggestions": [],
        "conversation_id": conversation_id,
    }
