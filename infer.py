"""تنبؤ بـ numpy فقط (بدون torch): أخف بكثير للاستضافة المجانية. يطابق model.py."""
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import config
from chem import atom_features, bond_features, BOND_DIM, canonical, physchem, lipinski_violations

NPZ_PATH = next((p for p in (config.BASE_DIR / "molsolve_np.npz", config.CKPT_DIR / "molsolve_np.npz") if p.exists()), config.BASE_DIR / "molsolve_np.npz")


def silu(x): return x / (1 + np.exp(-x))


def layernorm(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def lin(x, W, b): return x @ W.T + b


class NPModel:
    def __init__(self, path=NPZ_PATH):
        z = np.load(path, allow_pickle=False)
        self.p = {k: z[k] for k in z.files if not k.startswith("_")}
        self.tasks = [str(t) for t in z["_tasks"]]
        self.scaler = tuple(float(v) for v in z["_scaler"])
        self.layers = int(z["_layers"])
        self.test_metrics = json.loads(str(z["_test_metrics"]))
        self.dropout = config.DROPOUT

    def _drop(self, x, rng, train):
        if not train:
            return x
        mask = rng.random(x.shape) >= self.dropout
        return x * mask / (1 - self.dropout)

    def forward(self, g, rng=None, train=False):
        p = self.p
        h = lin(g["x"], p["atom_enc.weight"], p["atom_enc.bias"])
        src, dst = g["edge_index"]
        for i in range(self.layers):
            pre = f"layers.{i}."
            e = lin(g["edge_attr"], p[pre + "edge.weight"], p[pre + "edge.bias"])
            msg = silu(h[src] + e) if len(src) else np.zeros_like(h)
            agg = np.zeros_like(h)
            if len(src):
                np.add.at(agg, dst, msg)
            z = (1 + p[pre + "eps"][0]) * h + agg
            z = lin(silu(lin(z, p[pre + "mlp.0.weight"], p[pre + "mlp.0.bias"])),
                    p[pre + "mlp.2.weight"], p[pre + "mlp.2.bias"])
            z = silu(layernorm(z, p[pre + "norm.weight"], p[pre + "norm.bias"]))
            h = h + self._drop(z, rng, train)
        summ = h.sum(0, keepdims=True)
        mean = summ / h.shape[0]
        fp = self._drop(silu(lin(g["fp"][None], p["fp_enc.0.weight"], p["fp_enc.0.bias"])), rng, train)
        zc = np.concatenate([mean, summ / 10.0, fp], axis=1)
        o = lin(zc, p["head.0.weight"], p["head.0.bias"])
        o = silu(layernorm(o, p["head.1.weight"], p["head.1.bias"]))
        o = self._drop(o, rng, train)
        return lin(o, p["head.4.weight"], p["head.4.bias"])[0]


def graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    x = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    src, dst, ea = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b); src += [i, j]; dst += [j, i]; ea += [f, f]
    ea = np.array(ea, dtype=np.float32) if ea else np.zeros((0, BOND_DIM), np.float32)
    fp = np.array(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=config.FP_BITS)), dtype=np.float32)
    return {"x": x, "edge_index": np.array([src, dst], dtype=np.int64).reshape(2, -1),
            "edge_attr": ea, "fp": fp}


def load():
    return NPModel()


def predict(smiles, model, passes=30, seed=0):
    c = canonical(smiles)
    if c is None:
        raise ValueError("SMILES غير صالح")
    g = graph(c)
    rng = np.random.default_rng(seed)
    runs = np.stack([model.forward(g, rng, train=True) for _ in range(passes)])
    mean_, std_ = model.scaler
    out = {"smiles": c, "physchem": physchem(c), "tasks": {}}
    out["lipinski_violations"] = lipinski_violations(out["physchem"])
    n_reg = len(config.REG_TASKS)
    for i, t in enumerate(model.tasks):
        v = runs[:, i]
        if i < n_reg:
            v = v * std_ + mean_
            out["tasks"][t] = {"type": "regression", "value": float(v.mean()), "uncertainty": float(v.std())}
        else:
            pr = 1 / (1 + np.exp(-v))
            out["tasks"][t] = {"type": "classification", "prob": float(pr.mean()), "uncertainty": float(pr.std())}
    return out
