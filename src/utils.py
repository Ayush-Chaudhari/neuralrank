import yaml
import os
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent

def load_config(config_path=None):
    if config_path is None:
        config_path = ROOT_DIR / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_logger(name: str):
    os.makedirs(ROOT_DIR / "logs", exist_ok=True)
    logger.add(
        ROOT_DIR / "logs" / f"{name}.log",
        rotation="10 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"
    )
    return logger