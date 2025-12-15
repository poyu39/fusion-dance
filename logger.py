import logging
import os
import sys


class RelativePathFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, project_root=None):
        super().__init__(fmt, datefmt)
        self.project_root = project_root or os.getcwd()
    
    def format(self, record):
        if hasattr(record, 'pathname'):
            try:
                record.relative_path = os.path.relpath(record.pathname, self.project_root)
            except ValueError:
                record.relative_path = os.path.basename(record.pathname)
        else:
            record.relative_path = 'unknown'
        
        return super().format(record)


def setup_logger(name, project_root, log_file=None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    
    formatter = RelativePathFormatter(
        "%(asctime)s %(relative_path)s:%(lineno)d %(name)s %(levelname)s: %(message)s",
        project_root=project_root
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if log_file:
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger