"""
logger_utils.py - Timestamped text-file logging for the climate-downscaling
pipeline.

Usage
-----
    from logger_utils import setup_logger

    logger, log_path = setup_logger(out_dir)
    logger.info("Starting pipeline")
    ...

Every call to `logger.info/warning/error(...)` is written to a plain text
file (one line per event, timestamped) inside the pipeline's output
directory, and is also echoed to the console so behaviour in a notebook or
terminal is unchanged.
"""
import logging
import os
from datetime import datetime

LOGGER_NAME = 'climate_downscaling'


def setup_logger(output_dir, log_filename=None, name=LOGGER_NAME, level=logging.INFO):
    """
    Configure a logger that writes timestamped entries to a text file inside
    `output_dir` and also prints them to the console.

    Parameters
    ----------
    output_dir : str
        Directory the log file will be written into (created if missing).
    log_filename : str, optional
        Name of the log file. Defaults to 'pipeline_log_<YYYYmmdd_HHMMSS>.txt'
        so re-running the pipeline never overwrites a previous run's log.
    name : str
        Logger name (use the default unless you need multiple independent
        loggers in the same process).
    level : int
        Logging level, e.g. logging.INFO or logging.DEBUG.

    Returns
    -------
    (logging.Logger, str)
        The configured logger and the full path to the log file.
    """
    os.makedirs(output_dir, exist_ok=True)
    if log_filename is None:
        log_filename = f"pipeline_log_{datetime.now():%Y%m%d_%H%M%S}.txt"
    log_path = os.path.join(output_dir, log_filename)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # avoid duplicate lines if setup_logger is called twice
    logger.propagate = False

    fmt = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    logger.info(f"Log file created at: {log_path}")
    return logger, log_path
