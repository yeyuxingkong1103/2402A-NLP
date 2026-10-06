"""日志工具模块"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

from src.config import config


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径
    
    Returns:
        配置好的logger对象
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    logger.setLevel(config.get('logging.level', 'INFO'))
    
    # 日志格式
    log_format = logging.Formatter(
        config.get('logging.format', 
                   '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=config.get('logging.max_bytes', 10 * 1024 * 1024),
            backupCount=config.get('logging.backup_count', 5),
            encoding='utf-8'
        )
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    else:
        # 默认日志文件
        default_log = config.get('logging.file', 'logs/app.log')
        log_path = Path(default_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            default_log,
            maxBytes=config.get('logging.max_bytes', 10 * 1024 * 1024),
            backupCount=config.get('logging.backup_count', 5),
            encoding='utf-8'
        )
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    
    return logger


# 默认日志记录器
logger = setup_logger('rag_roleplay')
