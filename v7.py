"""V7 additions: batch data audit, candidate rationales, batch uncertainty report, scope note, baseline table."""
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
import v6, dossier

SCOPE = {"ar": "حدود الصلاحية: هذه التنبؤات تصلح للتعليم والفرز الأولي فقط. لا تصلح لقرار علاجي أو دوائي أو لسلامة الاستخدام، ولا تغني عن قياس مخبري. كل رقم هنا تقدير من نموذج مدرَّب على بيانات عامة محدودة، وتقل دقته للجزيئات البعيدة عن بيانات التدريب.",
         "en": "Scope of validity: these predictions are for teaching and first-pass screening only. They are not for treatment, drug or safety decisions and do not replace lab measurements. Every number is an estimate from a model trained on limited public data, and it is less accurate for molecules far from the training data."}


def audit(raw):
    """Data audit on raw SMILES strings before any prediction."""
    invalid, seen, dups, salts, ood, edge = [], {}, [], [], [], []
    heavy = {}
    for s in raw:
        m = Chem.MolFromSmiles(s) if s else None
        if m is None or m.GetNumAtoms() == 0:
            invalid.append(s); continue
        c = Chem.MolToSmiles(m)
        if c in seen:
            dups.append(s); continue
        seen[c] = s
        if len(Chem.GetMolFrags(m)) > 1:
            salts.append(s)
        heavy[c] = m.GetNumHeavyAtoms()
        b = v6.domain(c)["band"]
        (ood if b == "out" else edge if b == "edge" else []).append(s) if b != "in" else None
    sizes = np.array(list(heavy.values())) if heavy else np.array([0])
    q1, q3 = np.percentile(sizes, [25, 75]); iqr = max(q3 - q1, 1)
    odd = [seen[c] for c, h in heavy.items() if h < 5 or h > 70 or (len(heavy) >= 8 and (h > q3 + 3 * iqr or h < q1 - 3 * iqr))]
    return {"total": len(raw), "valid_unique": len(seen), "invalid": invalid, "duplicates": dups, "salts_mixtures": salts,
            "out_of_domain": ood, "edge_of_domain": edge, "odd_size": odd}


def audit_rows(a, ar=True):
    n = lambda x: len(x)
    rows = [("إجمالي المدخلات" if ar else "Total inputs", a["total"], []),
            ("جزيئات صالحة وفريدة" if ar else "Valid unique molecules", a["valid_unique"], []),
            ("صيغ غير صالحة (ستُستبعد)" if ar else "Invalid SMILES (will be skipped)", n(a["invalid"]), a["invalid"]),
            ("مكررة" if ar else "Duplicates", n(a["duplicates"]), a["duplicates"]),
            ("أملاح أو مخاليط (أكثر من جزء)" if ar else "Salts or mixtures (multiple fragments)", n(a["salts_mixtures"]), a["salts_mixtures"]),
            ("خارج نطاق النموذج" if ar else "Outside model domain", n(a["out_of_domain"]), a["out_of_domain"]),
            ("على حافة النطاق" if ar else "Edge of domain", n(a["edge_of_domain"]), a["edge_of_domain"]),
            ("حجم شاذ" if ar else "Unusual size", n(a["odd_size"]), a["odd_size"])]
    return [{("البند" if ar else "item"): r[0], ("العدد" if ar else "count"): r[1], ("أمثلة" if ar else "examples"): "، ".join(x[:30] for x in r[2][:3])} for r in rows]


def rank_candidates(C, model, lang="ar"):
    """C = [(name, r)]. Rank with an explicit rule and give strengths and limits per molecule."""
    ar = lang == "ar"
    out = []
    for n, r in C:
        d = dossier.build(r, model, None, lang)
        lv = [a["level"] for a in d["axes"]]
        reds, yel = lv.count("red"), lv.count("yellow")
        tt = v6.trust_table(r, r["ad"], model.test_metrics, lang)
        hi = sum(1 for x in tt if x[1] == "high")
        good = [a["axis"] for a in d["axes"] if a["level"] == "green"]
        bad = [f"{a['axis']} ({dossier.ICON[a['level']]})" for a in d["axes"] if a["level"] in ("red", "yellow")]
        if r["alerts"]:
            bad.append(("تنبيهات هيكلية: " if ar else "structural alerts: ") + ", ".join(r["alerts"]))
        if r["tox_flags"]:
            bad.append(("إشارات سمّية: " if ar else "toxicity signals: ") + ", ".join(map(str, r["tox_flags"]))[:80])
        out.append({"name": n, "reds": reds, "yellows": yel, "trust_high": hi, "conf": r["confidence"], "good": good, "bad": bad,
                    "alerts": len(r["alerts"]), "tox": len(r["tox_flags"])})
    out.sort(key=lambda x: (x["reds"], x["alerts"] + x["tox"], x["yellows"], -x["trust_high"], -x["conf"]))
    return out


RULE = {"ar": "قاعدة الترتيب (مكتوبة ولا تخفي شيئاً): الأقل محاور حمراء أولاً، ثم الأقل تنبيهات وإشارات سمّية، ثم الأقل محاور صفراء، ثم الأكثر تنبؤات «أثق بها»، ثم الأعلى ثباتاً. هذا ترتيب للفرز والتعلم وليس حكماً علمياً نهائياً.",
        "en": "Ranking rule (stated, nothing hidden): fewest red axes first, then fewest alerts and toxicity signals, then fewest yellow axes, then most 'trust' predictions, then highest stability. This is for screening and learning, not a final scientific verdict."}


def uncertainty_report(res, model, lang="ar"):
    ar = lang == "ar"
    names = list(res[0]["tasks"]) if res else []
    cnt = {t: {"high": 0, "mid": 0, "low": 0} for t in names}
    bands = {"in": 0, "edge": 0, "out": 0}
    for r in res:
        bands[r["ad"]["band"]] += 1
        for t, lvl, _ in v6.trust_table(r, r["ad"], model.test_metrics, lang):
            cnt[t][lvl] += 1
    n = max(len(res), 1)
    rows = [{("الاختبار" if ar else "endpoint"): t, ("أثق به %" if ar else "trust %"): round(100 * c["high"] / n),
             ("بحذر %" if ar else "careful %"): round(100 * c["mid"] / n), ("لا أثق به %" if ar else "don't trust %"): round(100 * c["low"] / n)}
            for t, c in cnt.items()]
    tot = sum(sum(c.values()) for c in cnt.values())
    below = sum(c["mid"] + c["low"] for c in cnt.values())
    return {"n": len(res), "below_trust_pct": round(100 * below / max(tot, 1)), "domain": bands,
            "mean_stability": round(float(np.mean([r["confidence"] for r in res])), 1) if res else None, "rows": rows}


def baseline(lang="ar"):
    p = Path(__file__).resolve().parent / "baseline.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


BASE_NOTE = {"ar": "قارنّا بخطوط أساس بسيطة (انحدار ريدج على واصفات RDKit، وغابة عشوائية على الواصفات وبصمات مورغان) على تقسيم سقالات ثابت 80/10/10. تنبيه صريح: هذا ليس نفس مجموعة اختبار نموذجنا، لأن ملفات التدريب الأصلية وتقسيمها غير محفوظة في المستودع، فالمقارنة تقريبية وليست إثباتاً. والنتيجة: لا نستطيع ادعاء أن نموذجنا يتفوق على خط أساس بسيط. في الذوبانية حقق خط الأساس R² أعلى من رقمنا المعلن، وفي Tox21 المتوسطان متقاربان، وفي عبور الدماغ رقمنا المعلن (0.96) أعلى بكثير من خط الأساس على تقسيمي (0.75) لكن اختلاف التقسيم يمنع الجزم. يلزم إعادة تدريب نموذجنا على التقسيم نفسه للمقارنة العادلة.",
             "en": "We compared with simple baselines (ridge on RDKit descriptors, random forest on descriptors plus Morgan fingerprints) on a fixed 80/10/10 scaffold split. Honest caveat: this is not our model's own test set, because the original training files and split are not stored in the repository, so the comparison is approximate and not proof. Result: we cannot claim our model beats a simple baseline. On solubility the baseline scores a higher R² than our reported number, on Tox21 the means are close, and on brain penetration our reported 0.96 is far above the baseline's 0.75 on my split, but the different splits prevent a firm conclusion. A fair comparison needs retraining our model on the same split."}
