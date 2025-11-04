import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime
# 日志目录
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_logger(name: str = "backend") -> logging.Logger:
    """
    获取一个按天生成日志文件的 logger。
    每天生成一个新的日志文件，例如 logs/2025-11-03.log。
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # 避免重复添加 handler（防止多次导入模块时日志重复）
        return logger

    logger.setLevel(logging.INFO)

    # 日志文件名：logs/YYYY-MM-DD.log
    log_filename = LOG_DIR / f"{datetime.now():%Y-%m-%d}.log"

    # 按天轮转日志文件，保留 7 天
    file_handler = TimedRotatingFileHandler(
        filename=log_filename,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
        delay=False,
    )
    file_handler.suffix = "%Y-%m-%d.log"

    # 控制台输出
    console_handler = logging.StreamHandler()

    # 日志格式
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加 handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
