import logging
import os


def get_logger(name=__name__, level=logging.INFO):
    """
    获取日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 检查是否已经有处理器
    if not logger.handlers:
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(console_handler)
    
    return logger


# 创建默认日志记录器
logger = get_logger()
