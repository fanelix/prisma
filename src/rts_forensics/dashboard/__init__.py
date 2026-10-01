"""
rts_forensics.dashboard — thin Streamlit layer over the public pipeline.

This package is importable without Streamlit installed: the page registry is
plain data and Streamlit is imported lazily inside the renderer functions in
:mod:`rts_forensics.dashboard.views`. The analytical core never imports
Streamlit; the dashboard is only an adapter around
:func:`rts_forensics.pipeline.run_analysis` and the exporters
(:mod:`rts_forensics.report`, :mod:`rts_forensics.gis`), so the CLI and the
dashboard share one set of algorithms and one audit trail.

The six pages are fixed by the project specification and the repository design
(candidate clusters, station frame, prism series, map, evidence and downloads).
Their keys are stable identifiers; titles are what the sidebar shows.
"""

from __future__ import annotations

#: Ordered page registry: ``key`` is a stable identifier, ``title`` is UI text.
PAGES: tuple[dict[str, str], ...] = (
    {"key": "upload_audit", "title": "Upload & audit"},
    {"key": "station_frame", "title": "Station frame"},
    {"key": "prism_explorer", "title": "Prism explorer"},
    {"key": "map", "title": "Map"},
    {"key": "events", "title": "Events & investigations"},
    {"key": "downloads", "title": "Downloads"},
)

#: Convenience tuple of the six keys, in page order.
PAGE_KEYS: tuple[str, ...] = tuple(page["key"] for page in PAGES)

__all__ = ["PAGES", "PAGE_KEYS"]
