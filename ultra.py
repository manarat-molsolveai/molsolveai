"""Ultra helpers: name lookup, known-drug neighbours, plain-language verdict, radar, HTML report."""
import html, json, urllib.parse, urllib.request
from rdkit import Chem, DataStructs, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pathlib
_LIBJ = json.loads((pathlib.Path(__file__).resolve().parent / "library.json").read_text())
LIB_META = {e["name"]: e for e in _LIBJ["entries"]}
LIB = {n: e["smiles"] for n, e in LIB_META.items()}
LIB_INFO = {"count": len(LIB), "source": _LIBJ["source"], "generated": _LIBJ["generated"]}
_fp = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
_LIBFP = {n: _fp(Chem.MolFromSmiles(s)) for n, s in LIB.items() if Chem.MolFromSmiles(s)}


def similar(smiles, k=3):
    q = _fp(Chem.MolFromSmiles(smiles))
    sims = sorted(((DataStructs.TanimotoSimilarity(q, f), n) for n, f in _LIBFP.items()), reverse=True)[:k]
    return [(n, round(s, 2)) for s, n in sims]


def lookup_name(name):
    """Name -> SMILES from PubChem. Returns (smiles, error)."""
    try:
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/%s/property/SMILES/JSON" % urllib.parse.quote(name.strip())
        with urllib.request.urlopen(url, timeout=10) as r:
            p = json.load(r)["PropertyTable"]["Properties"][0]
        return p.get("SMILES") or p.get("ConnectivitySMILES"), None
    except Exception as e:
        return None, str(e)[:80]


def verdict(r, lang):
    R, ts = r["rules"], r["tasks"]
    items = []  # (color, text)
    ar = lang == "ar"
    if R["lipinski_viol"] == 0 and R["qed"] >= 0.5:
        items.append(("green", "شبه دواء جيد (يطابق Lipinski وQED مرتفع)." if ar else "Good drug-likeness (passes Lipinski, high QED)."))
    elif R["lipinski_viol"] <= 1:
        items.append(("yellow", "شبه دواء متوسط." if ar else "Moderate drug-likeness."))
    else:
        items.append(("red", "شبه دواء ضعيف (مخالفات متعددة لقواعد الدواء الفموي)." if ar else "Poor drug-likeness (several oral-drug rule violations)."))
    s = ts["LogS"]["value"]
    if s > -4:
        items.append(("green", f"ذوبانية جيدة (LogS {s:.1f})." if ar else f"Good solubility (LogS {s:.1f})."))
    elif s > -6:
        items.append(("yellow", f"ذوبانية متوسطة (LogS {s:.1f})." if ar else f"Moderate solubility (LogS {s:.1f})."))
    else:
        items.append(("red", f"ذوبانية ضعيفة جداً (LogS {s:.1f})." if ar else f"Very poor solubility (LogS {s:.1f})."))
    b = ts["BBB"]["prob"]
    items.append(("green" if b > .5 else "yellow", (f"يُرجّح أن يعبر الحاجز الدموي الدماغي ({b*100:.0f}%)." if b > .5 else f"يُرجّح ألا يعبر الدماغ بسهولة ({b*100:.0f}%).") if ar
                  else (f"Likely crosses the blood-brain barrier ({b*100:.0f}%)." if b > .5 else f"Unlikely to cross the blood-brain barrier easily ({b*100:.0f}%).")))
    n = len(r["tox_flags"])
    items.append(("green" if n == 0 else "yellow" if n <= 2 else "red",
                  (("لا مؤشرات سمّية مرتفعة." if n == 0 else f"{n} من 12 اختبار سمّية فوق 50%: " + ", ".join(r["tox_flags"]))) if ar
                  else (("No high toxicity signals." if n == 0 else f"{n} of 12 toxicity assays above 50%: " + ", ".join(r["tox_flags"])))))
    if r["alerts"]:
        items.append(("yellow", ("تنبيهات بنيوية: " if ar else "Structural alerts: ") + ", ".join(r["alerts"])))
    colors = [c for c, _ in items]
    overall = "red" if colors.count("red") >= 2 or items[0][0] == "red" and n >= 3 else "green" if colors.count("green") >= 3 and "red" not in colors else "yellow"
    return overall, items


DOT = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
LIMITS = [("MW", 500), ("LogP", 5), ("TPSA", 140), ("HBD", 5), ("HBA", 10), ("RotB", 10)]


def radar(entries):
    """entries: list of (label, result). Values are shown relative to the oral-drug limit (1.0 = limit)."""
    ang = np.linspace(0, 2 * np.pi, len(LIMITS), endpoint=False).tolist(); ang += ang[:1]
    fig = plt.figure(figsize=(4.4, 4.4)); ax = fig.add_subplot(111, polar=True)
    for label, r in entries:
        R = r["rules"]
        vals = [R["mw"], max(R["logp"], 0), R["tpsa"], R["hbd"], R["hba"], R["rot"]]
        v = [min(a / l, 1.6) for a, (_, l) in zip(vals, LIMITS)]; v += v[:1]
        ax.plot(ang, v, linewidth=2, label=label); ax.fill(ang, v, alpha=0.12)
    ax.plot(np.linspace(0, 2 * np.pi, 100), [1] * 100, "r--", linewidth=1)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([n for n, _ in LIMITS]); ax.set_yticks([0.5, 1.0]); ax.set_yticklabels(["", "limit"]); ax.set_ylim(0, 1.6)
    if len(entries) > 1:
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
    fig.tight_layout()
    return fig


def html_report(results, lang, t, tox_tasks):
    rtl = lang == "ar"
    parts = [f"<html dir='{'rtl' if rtl else 'ltr'}'><head><meta charset='utf-8'><title>MolSolveAI</title><style>body{{font-family:sans-serif;max-width:820px;margin:auto;padding:16px}}"
             "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}.c{border:1px solid #ddd;border-radius:8px;padding:12px;margin:16px 0}</style></head><body>",
             f"<h1>MolSolveAI</h1><p><b>{html.escape(t['disc'])}</b></p>"]
    for r in results:
        R, ts = r["rules"], r["tasks"]
        ov, items = verdict(r, lang)
        parts.append(f"<div class='c'><h2>{DOT[ov]} <code>{html.escape(r['smiles'])}</code></h2><ul>" + "".join(f"<li>{DOT[c]} {html.escape(x)}</li>" for c, x in items) + "</ul>")
        parts.append(f"<p>MW {R['mw']:.1f} | LogP {R['logp']:.2f} | TPSA {R['tpsa']:.1f} | HBD {R['hbd']} | HBA {R['hba']} | RotB {R['rot']} | QED {R['qed']:.2f} | {t['conf']} {r['confidence']}%</p>")
        parts.append(f"<p>LogS {ts['LogS']['value']:.2f} ± {ts['LogS']['uncertainty']:.2f} | BBB {ts['BBB']['prob']*100:.0f}%</p>")
        parts.append("<table><tr><th>assay</th><th>%</th></tr>" + "".join(f"<tr><td>{k}</td><td>{ts[k]['prob']*100:.0f}</td></tr>" for k in tox_tasks) + "</table></div>")
    parts.append("</body></html>")
    return "".join(parts)
