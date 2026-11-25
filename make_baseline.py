"""Offline script: simple baselines on a deterministic scaffold split (80/10/10, largest scaffold groups to train).
Not run inside the app. Writes baseline.json."""
import json, numpy as np, pandas as pd, warnings
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
warnings.filterwarnings("ignore"); RDLogger.DisableLog("rdApp.*")
DN = [d[0] for d in Descriptors._descList]
def feats(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    d = np.array([f(m) for _, f in Descriptors._descList], dtype=float)
    fp = np.zeros(1024); 
    for k, v in AllChem.GetMorganFingerprint(m, 2).GetNonzeroElements().items(): fp[k % 1024] += v
    return np.nan_to_num(np.clip(d, -1e6, 1e6)), fp
def split(smis):
    g = defaultdict(list)
    for i, s in enumerate(smis):
        try: sc = MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)
        except Exception: sc = ""
        g[sc].append(i)
    sets = sorted(g.values(), key=lambda x: (len(x), x[0]), reverse=True)
    n = len(smis); tr, va, te = [], [], []
    for s in sets:
        if len(tr) + len(s) <= 0.8 * n: tr += s
        elif len(va) + len(s) <= 0.1 * n: va += s
        else: te += s
    return np.array(tr), np.array(te)
def load(df, col):
    X = [feats(s) for s in df[col]]; ok = [i for i, x in enumerate(X) if x is not None]
    return df.iloc[ok].reset_index(drop=True), np.array([X[i][0] for i in ok]), np.array([X[i][1] for i in ok])
out = {"split": "scaffold 80/10/10, deterministic (largest scaffold groups to train)", "tasks": {}}
e = pd.read_csv("/tmp/data/esol.csv"); e, D, F = load(e, "smiles"); y = e["measured log solubility in mols per litre"].values
tr, te = split(list(e["smiles"]))
sc = StandardScaler().fit(D[tr]); rd = RidgeCV(alphas=np.logspace(-1, 3, 12)).fit(sc.transform(D[tr]), y[tr]); p = rd.predict(sc.transform(D[te]))
rf = RandomForestRegressor(300, random_state=0, n_jobs=-1).fit(np.hstack([D[tr], F[tr]]), y[tr]); p2 = rf.predict(np.hstack([D[te], F[te]]))
out["tasks"]["LogS"] = {"n_test": int(len(te)), "ridge_descriptors": {"R2": r2_score(y[te], p), "RMSE": mean_squared_error(y[te], p) ** .5},
                        "random_forest": {"R2": r2_score(y[te], p2), "RMSE": mean_squared_error(y[te], p2) ** .5}}
b = pd.read_csv("/tmp/data/bbbp.csv"); b, D, F = load(b, "smiles"); y = b["p_np"].values
tr, te = split(list(b["smiles"]))
rf = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(np.hstack([D[tr], F[tr]]), y[tr])
out["tasks"]["BBB"] = {"n_test": int(len(te)), "random_forest": {"AUC": roc_auc_score(y[te], rf.predict_proba(np.hstack([D[te], F[te]]))[:, 1])}}
t = pd.read_csv("/tmp/data/tox21.csv.gz"); t, D, F = load(t, "smiles"); tr, te = split(list(t["smiles"]))
X = np.hstack([D, F]); aucs = {}
for c in [c for c in t.columns if c not in ("mol_id", "smiles")]:
    yy = t[c].values; m = ~np.isnan(yy); a = np.array([i for i in tr if m[i]]); b_ = np.array([i for i in te if m[i]])
    rf = RandomForestClassifier(200, random_state=0, n_jobs=-1, class_weight="balanced").fit(X[a], yy[a])
    aucs[c] = {"AUC": roc_auc_score(yy[b_], rf.predict_proba(X[b_])[:, 1])}
out["tasks"].update({k: {"random_forest": v} for k, v in aucs.items()})
out["tox21_mean_AUC_random_forest"] = float(np.mean([v["AUC"] for v in aucs.values()]))
json.dump(out, open("baseline.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out["tasks"].items() if k in ("LogS", "BBB")}, indent=1), out["tox21_mean_AUC_random_forest"])
