import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_browser_assets_include_every_forensic_module():
    text = (ROOT / "app/index.html").read_text()
    for path in (ROOT / "rts_forensics").glob("*.py"):
        assert f'"rts_forensics/{path.name}"' in text
    assert '"scipy"' in text and '"matplotlib"' in text
    workflow = (ROOT / ".github/workflows/pages.yml").read_text()
    assert "cp -r rts_forensics _site/rts_forensics" in workflow


def test_dashboard_default_is_raw_forensics():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app/app.py")).run(timeout=20)
    assert not app.exception
    assert app.sidebar.radio[0].value == "Raw-observation forensics"
    assert any("RTS observation forensics" in element.value for element in app.title)


def test_cli_help():
    completed = subprocess.run(
        [sys.executable, "-m", "rts_forensics", "run", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--config" in completed.stdout
    assert "--out" in completed.stdout


def test_repository_filenames_with_spaces_are_url_encoded():
    import ast
    from urllib.parse import quote

    source = ast.parse((ROOT / "app/app.py").read_text())
    function = next(
        node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_ambil_repo"
    )
    function.decorator_list = []
    namespace = {
        "_unduh": lambda url, **kwargs: url,
        "URL_REPO": "https://example.test/data/",
        "quote": quote,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "app/app.py", "exec"), namespace)
    assert namespace["_ambil_repo"]("HLO Sept 2026.csv").endswith("HLO%20Sept%202026.csv")


def test_local_golden_regression():
    data = os.environ.get("RTS_GOLDEN_DATA")
    values = os.environ.get("RTS_GOLDEN_VALUES")
    if not data or not values:
        pytest.skip(
            "Independent HLO raw/golden bundle not supplied; set RTS_GOLDEN_DATA and RTS_GOLDEN_VALUES"
        )
    from rts_forensics.config import load_config
    from rts_forensics.pipeline import run

    expected = json.loads(Path(values).read_text())
    assert expected.get("checks"), "Golden JSON must supply checks and explicit tolerances"
    result = run(data, load_config(overrides=expected.get("config", {})))
    for check in expected["checks"]:
        assert "tolerance" in check, "No invented golden tolerance"
        table = result[check["table"]]
        for column, value in check["where"].items():
            table = table[table[column] == value]
        assert len(table) == 1, check
        assert table.iloc[0][check["column"]] == pytest.approx(check["expected"], abs=check["tolerance"])
