"""Tracing hooks with JSON trace export.

Provides a minimal span context manager and utilities for exporting traces to JSON.
Can be augmented with LangSmith, Langfuse, or OpenTelemetry providers.
"""

import json
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

# Global trace collector for the current session
_active_spans: list[dict[str, Any]] = []


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager that records a span with timing and attributes.

    Spans are collected in-memory and can be exported to JSON via export_traces().
    """
    span_id = str(uuid.uuid4())[:8]
    started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()

    span: dict[str, Any] = {
        "span_id": span_id,
        "name": name,
        "attributes": attributes or {},
        "started_at": started_at,
        "ended_at": None,
        "duration_seconds": None,
        "status": "running",
    }
    _active_spans.append(span)

    logger.debug("Span started: %s (id=%s)", name, span_id)

    try:
        yield span
        span["status"] = "ok"
    except Exception as e:
        span["status"] = "error"
        span["error"] = str(e)
        raise
    finally:
        duration = perf_counter() - started
        span["duration_seconds"] = round(duration, 4)
        span["ended_at"] = datetime.now(timezone.utc).isoformat()
        logger.debug("Span ended: %s (id=%s, %.4fs, status=%s)", name, span_id, duration, span["status"])


def get_traces() -> list[dict[str, Any]]:
    """Return all collected spans."""
    return list(_active_spans)


def clear_traces() -> None:
    """Clear all collected spans."""
    _active_spans.clear()


def export_traces_to_json(filepath: Path | str | None = None) -> Path:
    """Export all collected traces to a JSON file.

    If no filepath is given, writes to logs/traces_{timestamp}.json.
    """
    if filepath is None:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = logs_dir / f"traces_{timestamp}.json"
    else:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

    trace_data = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "span_count": len(_active_spans),
        "spans": _active_spans,
    }

    filepath.write_text(json.dumps(trace_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Traces exported: %s (%d spans)", filepath, len(_active_spans))
    return filepath
