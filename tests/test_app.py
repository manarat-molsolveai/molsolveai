"""Smoke test of the real Streamlit app: every tab loads and the main flows run without errors."""
import pathlib
from streamlit.testing.v1 import AppTest

APP = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")


def run():
    return AppTest.from_file(APP, default_timeout=300).run()


def click(at, label):
    [b for b in at.button if b.label == label][0].click()
    at.run()


def test_loads_without_error():
    at = run()
    assert not at.exception
    assert len(at.tabs) >= 10


def test_single_analysis_shows_dossier():
    at = run(); click(at, "حلّل")
    assert not at.exception
    assert any("ملف الأدلة" in e.label for e in at.expander)


def test_batch_audit_and_uncertainty_report():
    at = run()
    at.text_area[0].set_value("CC(=O)Oc1ccccc1C(=O)O\nCC(=O)Oc1ccccc1C(=O)O\nxx\nCn1cnc2c1c(=O)n(C)c(=O)n2C"); at.run()
    click(at, "شغّل الدفعة")
    assert not at.exception
    heads = [s.value for s in at.subheader]
    assert any("تدقيق البيانات" in h for h in heads) and any("عدم اليقين" in h for h in heads)


def test_compare_shows_ranking_with_reasons():
    at = run(); click(at, "قارن")
    assert not at.exception
    assert any("الترتيب مع المبررات" in s.value for s in at.subheader)


def test_model_card_lists_library_and_baseline():
    at = run()
    heads = [s.value for s in at.subheader]
    assert any("مكتبة الأدوية المرجعية" in h for h in heads)
    assert any("خطوط أساس" in h for h in heads)
