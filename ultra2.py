"""Novel features: atom-level 'why' heatmap, goal-driven molecule optimizer, quiz facts."""
import io, random
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import config, infer, druglike

# ---- 1. Atom attribution by masking each atom in the graph branch --------------------
def atom_scores(smiles, model, task):
    """Positive = this atom pushes the predicted value UP. Approximate: masks atom features
    in the graph branch only (the fingerprint branch is left unchanged)."""
    g = infer.graph(smiles)
    idx = model.tasks.index(task)
    base = model.forward(g)[idx]
    out = []
    for i in range(g["x"].shape[0]):
        g2 = dict(g); x2 = g["x"].copy(); x2[i] = 0; g2["x"] = x2
        out.append(float(base - model.forward(g2)[idx]))
    a = np.array(out)
    m = np.abs(a).max() or 1.0
    return (a / m).tolist()


def heatmap_png(smiles, scores, size=(420, 320)):
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    cols, rad = {}, {}
    for i, s in enumerate(scores):
        if abs(s) < 0.12:
            continue
        a = min(1.0, abs(s))
        cols[i] = (1.0, 1 - 0.65 * a, 1 - 0.65 * a) if s > 0 else (1 - 0.65 * a, 1 - 0.55 * a, 1.0)
        rad[i] = 0.45
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=list(cols), highlightAtomColors=cols, highlightAtomRadii=rad)
    d.FinishDrawing()
    return d.GetDrawingText()

# ---- 2. Goal-driven optimizer --------------------------------------------------------
RXNS = [
 ("add F", "[c;H1:1]>>[c:1]F"), ("add Cl", "[c;H1:1]>>[c:1]Cl"), ("add methyl", "[c;H1:1]>>[c:1]C"),
 ("add OH", "[c;H1:1]>>[c:1]O"), ("add OMe", "[c;H1:1]>>[c:1]OC"), ("add NH2", "[c;H1:1]>>[c:1]N"),
 ("add CF3", "[c;H1:1]>>[c:1]C(F)(F)F"), ("CH to N (pyridine-like)", "[c;H1:1]>>[n:1]"),
 ("acid to methyl ester", "[C:2](=[O:3])[OX2H1]>>[C:2](=[O:3])OC"), ("acid to amide", "[C:2](=[O:3])[OX2H1]>>[C:2](=[O:3])N"),
 ("OH to OMe", "[OX2H1;$(O[CX4]),$(Oc):1]>>[O:1]C"), ("CH3 to CF3", "[CH3:1]>>[C:1](F)(F)F"),
 ("Cl to F", "[Cl:1]>>F"), ("F to Cl", "[F:1]>>Cl"), ("NH2 to acetamide", "[NX3H2:1]>>[N:1]C(C)=O"),
 ("OH to F", "[OX2H1;$(O[CX4]),$(Oc):1]>>[F]"), ("add OH on chain", "[CX4;H2,H3:1]>>[C:1]O"),
 ("add methyl on chain", "[CX4;H2:1]>>[C:1]C"),
]
_R = [(n, AllChem.ReactionFromSmarts(s)) for n, s in RXNS]


def analogs(smiles, limit=80):
    mol = Chem.MolFromSmiles(smiles)
    seen = {Chem.MolToSmiles(mol)}
    out = []
    for name, rxn in _R:
        try:
            prods = rxn.RunReactants((mol,))
        except Exception:
            continue
        for ps in prods:
            try:
                p = ps[0]; Chem.SanitizeMol(p)
                s = Chem.MolToSmiles(p)
            except Exception:
                continue
            if s in seen or Chem.MolFromSmiles(s) is None:
                continue
            seen.add(s); out.append((name, s))
            if len(out) >= limit:
                return out
    return out


GOALS = {
 "sol": lambda r: r["tasks"]["LogS"]["value"],
 "tox": lambda r: -len(r["tox_flags"]) - sum(r["tasks"][k]["prob"] for k in config.TOX21_TASKS) / 12,
 "bbb_up": lambda r: r["tasks"]["BBB"]["prob"],
 "bbb_down": lambda r: -r["tasks"]["BBB"]["prob"],
 "qed": lambda r: r["rules"]["qed"],
}


def optimize(smiles, analyze, goal, top=5):
    base = analyze(smiles)
    f = GOALS[goal]
    b = f(base)
    cands = []
    for name, s in analogs(smiles):
        r = analyze(s, 12)
        if r:
            cands.append((f(r) - b, name, s, r))
    cands.sort(key=lambda x: -x[0])
    return base, cands[:top]

# ---- 3. Quiz -------------------------------------------------------------------------
def quiz_question(lib, rng=random):
    name = rng.choice(list(lib))
    m = Chem.MolFromSmiles(lib[name])
    R = druglike.rules(lib[name])
    kind = rng.choice(["mw", "logp", "lip", "hbd"])
    if kind == "mw":
        thr = 300; truth = R["mw"] > thr
        return name, lib[name], "mw", thr, truth, f"MW = {R['mw']:.0f}"
    if kind == "logp":
        thr = 2; truth = R["logp"] > thr
        return name, lib[name], "logp", thr, truth, f"LogP = {R['logp']:.2f}"
    if kind == "hbd":
        thr = 2; truth = R["hbd"] > thr
        return name, lib[name], "hbd", thr, truth, f"HBD = {R['hbd']}"
    truth = R["lipinski_viol"] == 0
    return name, lib[name], "lip", 0, truth, f"Lipinski violations = {R['lipinski_viol']}"
