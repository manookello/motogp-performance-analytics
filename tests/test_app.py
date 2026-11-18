from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "predictions.parquet"


@pytest.mark.skipif(not DATA.exists(), reason="processed data not built")
def test_app_runs_without_errors():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 5
