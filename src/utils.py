import logging
import os

from src.config import config as c


def ensure_dirs():
    """Create output folders if they don't exist yet."""
    for path in [c.paths.submission_dir, c.paths.model_dir, c.paths.log_dir]:
        os.makedirs(path, exist_ok=True)


def setup_logging():
    """Configure logging to both a file and stdout."""
    log_file = os.path.join(c.paths.log_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
