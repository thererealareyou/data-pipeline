# logging_config.py
import logging
import sys
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: str = "app.log") -> None:
    """Единая точка настройки логгирования для всего проекта"""

    # Создаём директорию для логов
    Path("logs").mkdir(exist_ok=True)

    # Конфигурация
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(name)-2s | %(levelname)-2s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"logs/{log_file}", encoding="utf-8")
        ]
    )

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)