from loguru import logger

def setup_logger(log_file: str, log_level: int):
    logger.remove()
    if log_level == 0:
        return
    logger.add(log_file, level="INFO")
