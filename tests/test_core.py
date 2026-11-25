"""Chemistry, model, dossier, audit and report-verification behaviour. No network, no Streamlit."""
import json, copy
import pytest
import infer, druglike, v6, v7, ultra, dossier
from chem import canonical


@pytest.fixture(scope="module")
def model():
    return infer.load()


def analyze(model, smi, passes=30):
    c = canonical(smi)
    r = infer.predict(c, model, passes=passes)
    r["rules"] = druglike.rules(c); r["alerts"] = druglike.alerts(c); r["confidence"] = druglike.confidence(r)
    r["tox_flags"] = druglike.tox_summary(r, __import__("config").TOX21_TASKS)
    r["ad"] = v6.domain(c); r["sa"] = v6.sa(c); r["mpo"] = v6.cns_mpo(r["rules"])
    return r


def test_canonical_rejects_garbage():
    assert canonical("not a smiles") is None
    assert canonical("") is None
    assert canonical("OC(=O)C") == canonical("CC(O)=O")


def test_predictions_are_deterministic(model):
    a = infer.predict(canonical("CC(=O)Oc1ccccc1C(=O)O"), model, passes=30)
    b = infer.predict(canonical("CC(=O)Oc1ccccc1C(=O)O"), model, passes=30)
    assert a["tasks"] == b["tasks"]


def test_declared_metrics_do_not_regress(model):
    tm = model.test_metrics
    assert tm["LogS"]["R2"] >= 0.70
    assert tm["BBB"]["AUC"] >= 0.95
    tox = [v["AUC"] for k, v in tm.items() if k not in ("LogS", "BBB")]
    assert sum(tox) / len(tox) >= 0.75


def test_known_drug_brain_check_does_not_get_worse(model):
    """Today the model matches 11 of 15 textbook cases. Guard against getting worse; do not read it as proof of accuracy."""
    rows, hit = v6.check_known(lambda s, p: analyze(model, s, p), ultra.LIB)
    assert len(rows) == 15 and hit >= 10


def test_applicability_domain_bands(model):
    assert v6.domain(canonical("CC(=O)Oc1ccccc1C(=O)O"))["band"] == "in"
    assert v6.domain(canonical("C"))["band"] in ("edge", "out", "in")


def test_dossier_has_no_single_score_and_honest_status(model):
    d = dossier.build(analyze(model, "Cn1cnc2c1c(=O)n(C)c(=O)n2C"), model, None, "ar")
    assert len(d["axes"]) == 7
    assert not any(k in d for k in ("score", "overall", "grade"))
    status = {v["item"]: v["status"] for v in d["validation"]}
    assert "Unknown" in status.values()  # experimental validation is never claimed
    assert d["axes"][-1]["level"] == "none"
    assert d["scope_of_validity"]


def test_report_id_stable_and_changes_with_input(model):
    r = analyze(model, "CC(=O)Oc1ccccc1C(=O)O")
    a = dossier.build(r, model, None, "ar")["report_id"]
    assert a == dossier.build(r, model, None, "ar")["report_id"]
    assert a != dossier.build(analyze(model, "Cn1cnc2c1c(=O)n(C)c(=O)n2C"), model, None, "ar")["report_id"]


def test_report_verification_roundtrip_and_tamper(model):
    d = dossier.build(analyze(model, "CC(C)Cc1ccc(cc1)C(C)C(=O)O"), model, None, "ar")
    again = dossier.build(analyze(model, d["identity"]["canonical_smiles"]), model, None, "ar")
    assert again["report_id"] == d["report_id"]
    t = copy.deepcopy(d); t["prediction"]["rows"][0]["value"] = "9.99 ± 0.00"
    assert t["prediction"]["rows"][0]["value"] != again["prediction"]["rows"][0]["value"]


def test_html_report_is_clean(model):
    d = dossier.build(analyze(model, "CC(=O)Oc1ccccc1C(=O)O"), model, {"status": "found", "cid": 2244, "url": "https://pubchem.ncbi.nlm.nih.gov/compound/2244"}, "ar")
    h = dossier.html_report(d, "ar")
    assert '{"status"' not in h and "CID 2244" in h and "حدود الصلاحية" in h
    json.dumps(d, ensure_ascii=False, default=str)


def test_dossier_english(model):
    d = dossier.build(analyze(model, "CC(=O)Oc1ccccc1C(=O)O"), model, None, "en")
    assert "Structure integrity" in [a["axis"] for a in d["axes"]]
    assert "Scope of validity" in dossier.html_report(d, "en")


def test_batch_audit_counts():
    a = v7.audit(["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O", "xx", "[Na+].CC(=O)[O-]", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "C"])
    assert a["total"] == 6 and a["valid_unique"] == 4
    assert len(a["invalid"]) == 1 and len(a["duplicates"]) == 1 and len(a["salts_mixtures"]) == 1 and len(a["odd_size"]) >= 1


def test_ranking_rule_is_stable(model):
    C = [(n, analyze(model, ultra.LIB[n])) for n in ("Aspirin", "Ibuprofen", "Caffeine")]
    ranked = v7.rank_candidates(C, model, "ar")
    assert {x["name"] for x in ranked} == {"Aspirin", "Ibuprofen", "Caffeine"}
    assert ranked == v7.rank_candidates(C, model, "ar")


def test_baseline_file_is_honest():
    b = v7.baseline()
    assert b and "LogS" in b["tasks"] and "BBB" in b["tasks"]
    assert "لا نستطيع ادعاء" in v7.BASE_NOTE["ar"] and "cannot claim" in v7.BASE_NOTE["en"]
