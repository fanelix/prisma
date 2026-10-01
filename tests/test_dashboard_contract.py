"""
test_dashboard_contract.py — the dashboard is a thin, auditable adapter.

These tests pin the interface the rest of the project (and the CI environment)
relies on:

* ``rts_forensics.dashboard`` imports without Streamlit installed — CI installs
  only ``.[dev,report]``, so a top-level Streamlit import would break the suite;
* the six documented pages exist, with their exact keys and titles, in order;
* every page key has a renderer in ``views.PAGE_RENDERERS``;
* no core module imports Streamlit (the core never depends on the UI);
* the Streamlit import inside ``views`` is lazy (inside functions), not at
  module import time.

No production algorithm is exercised here; the numerical tests live in the
other test modules.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import rts_forensics
from rts_forensics import dashboard

EXPECTED_PAGES = (
    ("upload_audit", "Upload & audit"),
    ("station_frame", "Station frame"),
    ("prism_explorer", "Prism explorer"),
    ("map", "Map"),
    ("events", "Events & investigations"),
    ("downloads", "Downloads"),
)

_STREAMLIT_IMPORT = re.compile(
    r"^\s*(?:import\s+streamlit\b|from\s+streamlit\b)", re.MULTILINE
)
_TOP_LEVEL_STREAMLIT_IMPORT = re.compile(
    r"^(?:import\s+streamlit\b|from\s+streamlit\b)", re.MULTILINE
)


def test_pages_are_exactly_the_six_documented_pages_in_order():
    assert [(page["key"], page["title"]) for page in dashboard.PAGES] == [
        (key, title) for key, title in EXPECTED_PAGES
    ]


def test_page_keys_match_the_page_registry():
    assert dashboard.PAGE_KEYS == tuple(key for key, _ in EXPECTED_PAGES)
    assert len(set(dashboard.PAGE_KEYS)) == len(dashboard.PAGE_KEYS)


def test_every_page_has_a_renderer():
    from rts_forensics.dashboard import views

    assert set(views.PAGE_RENDERERS) == set(dashboard.PAGE_KEYS)
    for key, renderer in views.PAGE_RENDERERS.items():
        assert callable(renderer), f"renderer for {key} is not callable"


def test_core_modules_never_import_streamlit():
    """The analytical core must not depend on the UI framework."""
    package_root = Path(rts_forensics.__file__).resolve().parent
    offenders = []
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root)
        if "dashboard" in relative.parts:
            continue
        if _STREAMLIT_IMPORT.search(path.read_text(encoding="utf-8")):
            offenders.append(str(relative))
    assert not offenders, f"core modules import streamlit: {offenders}"


def test_dashboard_views_import_streamlit_lazily():
    from rts_forensics.dashboard import views

    source = Path(views.__file__).read_text(encoding="utf-8")
    assert not _TOP_LEVEL_STREAMLIT_IMPORT.search(source), (
        "views.py must import streamlit inside renderer functions, not at module level"
    )


def test_importing_the_dashboard_does_not_load_streamlit():
    """Run in a subprocess so an installed-but-mocked Streamlit cannot hide it."""
    code = (
        "import sys; "
        "import rts_forensics.dashboard; "
        "import rts_forensics.dashboard.views; "
        "assert 'streamlit' not in sys.modules, 'streamlit was imported eagerly'; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
