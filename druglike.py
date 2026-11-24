"""Drug-likeness, alerts and confidence helpers (RDKit, no ML)."""
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED, rdMolDescriptors, Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

_params = FilterCatalogParams()
_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
_cat = FilterCatalog(_params)


def rules(smiles):
    m = Chem.MolFromSmiles(smiles)
    mw, logp = Descriptors.MolWt(m), Crippen.MolLogP(m)
    hbd, hba = Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m)
    tpsa, rot = rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumRotatableBonds(m)
    lip = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    veber = int(rot > 10) + int(tpsa > 140)
    ghose = int(not (160 <= mw <= 480)) + int(not (-0.4 <= logp <= 5.6))
    return {"mw": mw, "logp": logp, "tpsa": tpsa, "hbd": hbd, "hba": hba, "rot": rot,
            "lipinski_viol": lip, "veber_viol": veber, "ghose_viol": ghose,
            "qed": float(QED.qed(m)), "heavy": m.GetNumHeavyAtoms(),
            "rings": rdMolDescriptors.CalcNumRings(m)}


def alerts(smiles):
    m = Chem.MolFromSmiles(smiles)
    return sorted({e.GetDescription() for e in _cat.GetMatches(m)})


def confidence(result):
    """Rough 0-100 score from ensemble spread across all tasks (higher = steadier)."""
    t = result["tasks"]
    sp = [v["uncertainty"] for k, v in t.items() if v["type"] == "classification"]
    reg = t.get("LogS", {}).get("uncertainty", 0.3)
    s = 1.0 - min(1.0, (sum(sp) / len(sp)) / 0.25 * 0.5 + min(reg, 1.0) * 0.5)
    return int(round(max(0.0, s) * 100))


def tox_summary(result, tox_tasks, thr=0.5):
    hits = [k for k in tox_tasks if result["tasks"][k]["prob"] >= thr]
    return hits
