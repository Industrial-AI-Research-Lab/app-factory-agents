"""Logging protection for secret checkpoint callback path segments."""

import logging
import re


_PATH = re.compile(r"(/api/checkpoints/[^/\s]+/)(.*?)(/points/?(?=[?\s\"']|$))")


class CheckpointNonceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Uvicorn's AccessFormatter unpacks these five arguments itself; retain
        # the structured record and redact the full_path value in place.
        if (
            record.name == "uvicorn.access"
            and isinstance(record.args, tuple)
            and len(record.args) == 5
        ):
            client, method, path, version, code = record.args
            if isinstance(path, str):
                record.args = (
                    client,
                    method,
                    _PATH.sub(r"\1[REDACTED]\3", path),
                    version,
                    code,
                )
            return True
        try:
            rendered = record.getMessage()
        except Exception:
            return True
        redacted = _PATH.sub(r"\1[REDACTED]\3", rendered)
        if redacted != rendered:
            record.msg = redacted
            record.args = ()
        return True


_filter = CheckpointNonceFilter()


def install_checkpoint_log_filter() -> None:
    """Install once on access/request loggers commonly containing URLs."""
    for name in (
        "uvicorn.access",
        "httpx",
        "httpcore",
        "httpcore.connection",
        "httpcore.connection_pool",
        "httpcore.http11",
        "httpcore.http2",
        "httpcore.proxy",
    ):
        logger = logging.getLogger(name)
        if not any(isinstance(item, CheckpointNonceFilter) for item in logger.filters):
            logger.addFilter(_filter)
