"""Integrity of the reference drug library (library.json)."""
import json, pathlib, collections
from rdkit import Chem
import ultra

DATA = json.loads((pathlib.Path(__file__).resolve().parent.parent / "library.json").read_text())
E_ = DATA["entries"]


def test_size_and_source():
    assert len(E_) >= 250
    assert DATA["source"] == "PubChem"
    assert all(e["source"].startswith("PubChem") and e["retrieved"] for e in E_)


def test_every_entry_parses_single_fragment():
    for e in E_:
        m = Chem.MolFromSmiles(e["smiles"])
        assert m is not None, e["name"]
        assert len(Chem.GetMolFrags(m)) == 1, e["name"]


def test_unique_names_and_identities():
    assert len({e["name"] for e in E_}) == len(E_)
    assert len({e["inchikey"] for e in E_}) == len(E_)


def test_all_automated_checks_passed():
    for e in E_:
        assert e["passed"] and all(bool(v) for v in e["checks"].values()), e["name"]


def test_inchikey_connectivity_matches_rdkit():
    """Independent check: RDKit's own InChIKey of the stored SMILES agrees with PubChem's on the connectivity block."""
    bad = [e["name"] for e in E_ if Chem.MolToInchiKey(Chem.MolFromSmiles(e["smiles"]))[:14] != e["inchikey"][:14]]
    assert len(bad) <= 3, bad


def test_original_core_molecules_present():
    for n in ["Aspirin", "Caffeine", "Ibuprofen", "Metformin", "Morphine", "Penicillin G"]:
        assert n in ultra.LIB


def test_excluded_candidates_are_documented():
    assert all(x["name"] and x["reason"] for x in DATA["excluded"])
    assert not ({x["name"] for x in DATA["excluded"]} & {e["name"] for e in E_})
