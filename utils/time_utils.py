from __future__ import annotations

from datetime import datetime, timedelta


def today_str() -> str:
    """Return today's local date as YYYY-MM-DD."""
    return datetime.now().date().isoformat()


def offset_date(days: int) -> str:
    """Return the local date offset by ``days`` as YYYY-MM-DD."""
    target_date = datetime.now().date() + timedelta(days=days)
    return target_date.isoformat()


def current_datetime_str() -> str:
    """Return a human-readable local datetime string for prompt injection."""
    now = datetime.now().astimezone()
    return now.strftime("%A, %d %B %Y, %H:%M %Z")
