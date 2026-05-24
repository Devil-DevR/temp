# ============================================================
# modules/logger.py
# Centralized Loguru logger for EmotiLearn
#
# Features:
#   - Colored console output per log level
#   - Rotating file logs (10 MB cap, 7-day retention)
#   - Separate error log file for quick debugging
#   - Structured format: timestamp | level | module | message
#   - One import line in every other module:
#       from modules.logger import logger
# ============================================================

import sys
from pathlib import Path
from loguru import logger

# ── Log directory ─────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── Remove default Loguru sink (we define our own) ────────────
logger.remove()

# ── Console sink — colored, human-readable ───────────────────
# Format explanation:
#   {time:HH:mm:ss}   → short timestamp  (14:03:27)
#   {level:<8}        → level left-padded (DEBUG   / INFO    / WARNING)
#   {name}            → module name       (modules.fusion)
#   {function}        → function name     (fuse)
#   {line}            → line number       (142)
#   {message}         → the log message
logger.add(
    sys.stderr,
    level="DEBUG",
    colorize=True,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    ),
)

# ── Main rotating file log ────────────────────────────────────
# Rotates at 10 MB, keeps 7 days of history, compresses old logs
logger.add(
    LOG_DIR / "emotilearn.log",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    encoding="utf-8",
    format=(
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level:<8} | "
        "{name}:{function}:{line} — "
        "{message}"
    ),
)

# ── Error-only file log ───────────────────────────────────────
# Separate file that only captures WARNING and above
# Useful for quickly spotting problems without noise
logger.add(
    LOG_DIR / "errors.log",
    level="WARNING",
    rotation="5 MB",
    retention="30 days",
    compression="zip",
    encoding="utf-8",
    format=(
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level:<8} | "
        "{name}:{function}:{line} — "
        "{message}"
    ),
)

# ── Named sub-loggers per module ──────────────────────────────
# These inherit all sinks above but carry the module name.
# Usage in any module:
#   from modules.logger import logger          # general
#   from modules.logger import audio_logger   # audio-specific
#
audio_logger  = logger.bind(module="audio")
text_logger   = logger.bind(module="text")
fusion_logger = logger.bind(module="fusion")
api_logger    = logger.bind(module="api")
stt_logger    = logger.bind(module="stt")

# ── Startup confirmation ──────────────────────────────────────
logger.debug(
    "Logger initialized | console=stderr | "
    f"file={LOG_DIR / 'emotilearn.log'} | "
    f"errors={LOG_DIR / 'errors.log'}"
)
