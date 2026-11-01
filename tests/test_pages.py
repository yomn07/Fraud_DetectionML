from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["overview", "performance", "simulator", "scoring", "insights", "about"]


@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_exception(page):
    at = AppTest.from_file(str(ROOT / "app_pages" / f"{page}.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_main_entry():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_scoring_sample_and_single_form():
    at = AppTest.from_file(str(ROOT / "app_pages" / "scoring.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Transactions" in m.label for m in at.metric)       # bundled sample scored in demo mode
    at.button[0].click().run()                                      # form submit
    assert not at.exception, [e.value for e in at.exception]
