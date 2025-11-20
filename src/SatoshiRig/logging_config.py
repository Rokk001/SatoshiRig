import logging
import os
from logging.handlers import RotatingFileHandler


def configure_logging(
    level: str = None, log_file: str = None, verbose: bool = False
) -> None:
    """Configure root logging for the application.

    - `level`: string like 'INFO' or 'DEBUG'. If None, will use env LOG_LEVEL or 'INFO'.
    - `log_file`: optional path for rotating file logging. If None, only stdout/stderr used.
    - `verbose`: if True, enables more verbose internal debug logs.
    """
    # Resolve defaults from environment
    level = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    # All logging now goes to stdout/stderr (Docker logs). Ignore any log_file value.
    if log_file or os.environ.get("LOG_FILE"):
        logging.getLogger("SatoshiRig.logging_config").debug(
            "File logging is disabled; all logs will be sent to stdout/stderr."
        )
    log_file = None

    level_const = getattr(logging, level, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level_const)

    # Remove existing file handlers if any (legacy compatibility)
    for h in list(root.handlers):
        if isinstance(h, RotatingFileHandler):
            try:
                h.close()
            except Exception:
                pass
            root.removeHandler(h)

    # Ensure a StreamHandler exists
    has_stream = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    if not has_stream:
        sh = logging.StreamHandler()
        sh.setLevel(level_const)
        sh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        root.addHandler(sh)

    # File handlers are intentionally not added to ensure Docker log visibility.

    # Optionally enable very verbose internal debug logs
    if verbose:
        logging.getLogger("SatoshiRig").setLevel(logging.DEBUG)
