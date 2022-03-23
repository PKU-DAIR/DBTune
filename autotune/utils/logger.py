import logging
import os
import pathlib

LOG_LEVEL_DICT = {
    "logging.debug": logging.DEBUG,
    "logging.info": logging.INFO,
    "logging.warning": logging.WARNING,
    "logging.error": logging.ERROR,
    "logging.critical": logging.CRITICAL
}

DEFAULT_LOG_FORMAT = os.environ.get('DEFAULT_LOG_FORMAT',
                                    '[%(asctime)s:%(filename)s#L%(lineno)d:%(levelname)s]: %(message)s')
DEFAULT_LOG_LEVEL = LOG_LEVEL_DICT[os.environ.get('DEFAULT_LOG_LEVEL', 'logging.INFO').lower()]


def get_logger(logger_name, log_file_path, log_level=DEFAULT_LOG_LEVEL, log_format=DEFAULT_LOG_FORMAT):
    pathlib.Path(log_file_path).parents[0].mkdir(exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    handler = logging.FileHandler(log_file_path)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(log_format))

    logger.addHandler(handler)
    return logger
