"""
app.py — Streamlit entry point for the rts-forensics dashboard.

Run with::

    pip install -e ".[dashboard]"
    streamlit run src/rts_forensics/dashboard/app.py

The dashboard is a thin adapter: it loads a configuration, uploads a CSV,
calls :func:`rts_forensics.pipeline.run_analysis` and renders one of the six
pages from :data:`rts_forensics.dashboard.PAGES`. It never re-implements an
algorithm. The core package does not import Streamlit.

Two notices are always visible: the section-8 limitations, and the statement
that results are exploratory while no TARP is configured. No slope is ever
declared safe or unsafe.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from rts_forensics import __version__, load_config, validate_config
from rts_forensics.config import is_not_configured
from rts_forensics.dashboard import PAGES
from rts_forensics.dashboard.views import PAGE_RENDERERS
from rts_forensics.pipeline import run_analysis

LIMITATIONS = (
    "No slope stability or factor of safety can be established from RTS data alone.",
    "No alarm status exists without a site TARP.",
    "No absolute movement can be established without external control: the frame "
    "is defined by the network.",
    "No causes can be attributed without rainfall, blasting, stacking or geology data.",
    "Resection references are not identifiable from the export.",
    "Unobserved periods are not evidence of stability.",
)

_TITLES = {page["key"]: page["title"] for page in PAGES}


def _save_upload(uploaded) -> Path:
    suffix = Path(uploaded.name).suffix or ".csv"
    handle = tempfile.NamedTemporaryFile(
        prefix="rts-forensics-upload-", suffix=suffix, delete=False
    )
    handle.write(uploaded.getvalue())
    handle.close()
    return Path(handle.name)


def _limitations_notice() -> None:
    st.divider()
    st.subheader("Limitations of RTS data")
    with st.expander("What this data cannot establish", expanded=True):
        for line in LIMITATIONS:
            st.markdown(f"- {line}")


def main() -> None:
    st.set_page_config(page_title="rts-forensics", layout="wide")
    st.title("rts-forensics")
    st.caption(
        f"Version {__version__} — auditable forensic analysis of monthly RTS "
        "prism-monitoring exports"
    )

    with st.sidebar:
        st.header("Run")
        config_path = st.text_input(
            "Configuration file",
            value="config.yaml",
            help="Path to config.yaml. Relative paths resolve from the directory "
            "streamlit was started in (the repository root by default).",
        )
        uploaded = st.file_uploader("Monthly RTS export (CSV)", type=["csv", "txt"])
        run_clicked = st.button("Run analysis", type="primary")
        st.divider()
        page_key = st.radio(
            "Page",
            options=[page["key"] for page in PAGES],
            format_func=lambda key: _TITLES[key],
        )

    run = st.session_state.get("run")
    config = st.session_state.get("config")
    audit = st.session_state.get("audit")

    if run_clicked:
        if uploaded is None:
            st.error("Upload a monthly RTS export before running the analysis.")
        else:
            try:
                config = load_config(config_path)
                audit = validate_config(config)
                source = _save_upload(uploaded)
                with st.spinner("Running the pipeline..."):
                    run = run_analysis(source, config=config)
                st.session_state["run"] = run
                st.session_state["config"] = config
                st.session_state["audit"] = audit
                if "rts_forensics_download_dir" in st.session_state:
                    del st.session_state["rts_forensics_download_dir"]
            except FileNotFoundError:
                st.error(f"Configuration file not found: {config_path}")
            except Exception as exc:  # surface parse/pipeline errors in the UI
                st.error(f"Analysis failed: {exc}")

    if run is None:
        st.info(
            "Upload a monthly RTS export and select 'Run analysis'. The run uses the "
            "same pipeline as the CLI, so both interfaces produce identical outputs."
        )
    else:
        PAGE_RENDERERS[page_key](run, config=config, audit=audit)

    if config is None or is_not_configured(getattr(config, "tarp", None)):
        st.warning(
            "Exploratory only: no TARP is configured, so no alarm state is produced "
            "and no slope is declared safe or unsafe. Screening labels are "
            "exploratory evidence, not site safety decisions."
        )
    _limitations_notice()


if __name__ == "__main__":
    main()
