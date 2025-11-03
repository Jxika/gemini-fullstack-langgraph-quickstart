import os
from logger import get_logger
logger=get_logger(__name__)

logger.info("Program started")
logger.warning("Low memory warning")
logger.error("Request failed")
logger.critical("critical")


def print_dir_tree(startpath, indent=''):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

# 使用示例
print_dir_tree("../backend")  # 当前目录
