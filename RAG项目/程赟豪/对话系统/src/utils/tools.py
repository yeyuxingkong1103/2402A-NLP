"""工具函数模块"""
import re
import hashlib
import json
from typing import Any, Dict, List, Optional
from datetime import datetime


def generate_id(text: str) -> str:
    """生成唯一ID"""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def sanitize_filename(filename: str) -> str:
    """清理文件名"""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """格式化时间戳"""
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def parse_json_safe(text: str) -> Dict[str, Any]:
    """安全解析JSON"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def truncate_text(text: str, max_length: int = 500) -> str:
    """截断文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def extract_numbers(text: str) -> List[float]:
    """提取文本中的数字"""
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    return [float(m) for m in matches]


def clean_text(text: str) -> str:
    """清理文本"""
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text)
    # 去除特殊字符
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text.strip()
