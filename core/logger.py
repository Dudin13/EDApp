import sys
from loguru import logger
from pathlib import Path

# Create logs directory
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configure logger
def setup_logger():
    # Remove default handler
    logger.remove()
    
    # Add stdout handler with colors
    logger.add(
        sys.stdout, 
        colorize=True, 
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file handler for rotating logs
    logger.add(
        LOG_DIR / "edudin_{time}.log",
        rotation="10 MB",
        retention="1 week",
        compression="zip",
        level="DEBUG"
    )

setup_logger()

# Export logger
__all__ = ["logger"]
