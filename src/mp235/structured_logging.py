"""Structured JSON prediction logging for MP235 Cold Storage Spoilage Prediction.

Provides a reusable standard Python logging configuration with a JSON formatter
outputting directly to stdout. Ensures each log line is valid single-line JSON,
excludes sensitive fields, includes execution timing and metadata, and avoids
duplicate handlers.
"""

from __future__ import annotations

import datetime
import json
import logging
import sys
from typing import Any, Mapping

# Explicitly exclude sensitive or personal customer/system attributes
FORBIDDEN_FIELDS = {
    "password",
    "secret",
    "token",
    "api_key",
    "customer_id",
    "customer_name",
    "user_id",
    "email",
    "phone",
    "ssn",
    "aadhar",
}


class PredictionJSONFormatter(logging.Formatter):
    """Custom logging formatter outputting a single valid JSON record per line."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "severity": record.levelname,
            "event": getattr(record, "event", record.getMessage()),
            "logger": record.name,
        }

        # Model version metadata if available
        model_version = getattr(record, "model_version", None)
        if model_version is not None:
            log_payload["model_version"] = str(model_version)

        # Execution duration in milliseconds
        duration_ms = getattr(record, "prediction_duration_ms", None)
        if duration_ms is not None:
            log_payload["prediction_duration_ms"] = round(float(duration_ms), 3)

        # Prediction value
        prediction = getattr(record, "prediction", None)
        if prediction is not None:
            log_payload["prediction"] = prediction

        # Append safe extra metadata if provided
        extra_data = getattr(record, "extra_payload", None)
        if isinstance(extra_data, Mapping):
            for k, v in extra_data.items():
                if k.lower() not in FORBIDDEN_FIELDS and k not in log_payload:
                    log_payload[k] = v

        return json.dumps(log_payload, default=str)


def get_prediction_logger(name: str = "mp235.prediction") -> logging.Logger:
    """Configures and returns a singleton logger with a JSON stdout handler.

    Guarantees:
    - Exactly one stdout StreamHandler (no duplicated logs across calls).
    - Output sent directly to sys.stdout, not to a file.
    - Propagation disabled so root loggers do not duplicate formatted records.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    has_stdout_handler = any(
        isinstance(h, logging.StreamHandler) and h.stream is sys.stdout
        for h in logger.handlers
    )

    if not has_stdout_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(PredictionJSONFormatter())
        logger.addHandler(handler)

    return logger


def log_prediction(
    logger: logging.Logger,
    prediction: float | int | list[float],
    duration_ms: float,
    model_version: str | None = "1.0",
    event_name: str = "prediction_success",
    extra_fields: Mapping[str, Any] | None = None,
) -> None:
    """Reusable prediction logging helper for API and inference services.

    FastAPI integration (in POST /predict):
        start = time.perf_counter()
        pred = model.predict(...)
        duration_ms = (time.perf_counter() - start) * 1000
        log_prediction(logger, prediction=pred, duration_ms=duration_ms, model_version="1.0")
    """
    safe_extras: dict[str, Any] = {}
    if extra_fields:
        for k, v in extra_fields.items():
            if k.lower() not in FORBIDDEN_FIELDS:
                safe_extras[k] = v

    logger.info(
        event_name,
        extra={
            "event": event_name,
            "prediction": prediction,
            "prediction_duration_ms": duration_ms,
            "model_version": model_version,
            "extra_payload": safe_extras,
        },
    )
