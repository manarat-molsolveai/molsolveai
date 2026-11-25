"""Evidence Dossier: per-molecule evidence file with explicit verification status, a multi-axis panel
(no single combined score) and a reproducibility record."""
import hashlib, html, json, datetime, urllib.request, urllib.parse
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
import config, v6, ultra
SCOPE_AR = "حدود الصلاحية: للتعليم والفرز الأولي فقط، لا لقرار علاجي أو دوائي أو لسلامة الاستخدام، ولا تغني عن قياس مخبري."
SCOPE_EN = "Scope of validity: teaching and first-pass screening only, not for treatment, drug or safety decisions, and no substitute for lab measurement."

APP_VERSION = "7.0"
_SHA = None


def model_sha():
    global _SHA
    if _SHA is None:
        _SHA = hashlib.sha256(open(config.BASE_DIR / "molsolve_np.npz", "rb").read()).hexdigest()
    return _SHA


def pubchem(smiles):
    """Look the structure up in PubChem by InChIKey. Returns dict(status, cid, url)."""
    try:
        key = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
        u = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/%s/cids/JSON" % urllib.parse.quote(key)
        with urllib.request.urlopen(u, timeout=5) as r:
            cid = json.load(r)["IdentifierList"]["CID"][0]
        return {"status": "found", "cid": cid, "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"}
    except urllib.error.HTTPError as e:
        return {"status": "not_found" if e.code == 404 else "unavailable"}
    except Exception:
        return {"status": "unavailable"}


# tags
CALC, PRED, UNV = "calc", "pred", "unverified"
TAG = {"ar": {CALC: "محسوب بدقة", PRED: "تنبؤ تقديري", UNV: "غير متحقق تجريبياً"},
       "en": {CALC: "computed exactly", PRED: "estimated prediction", UNV: "not experimentally verified"}}
COL = {"green": "#2e9e44", "yellow": "#e6a800", "red": "#d93025", "info": "#1a73e8", "none": "#9aa0a6"}
CRITERIA = {"ar": "معايير الألوان (تقديرية وضعناها بالاجتهاد، لم تُعايَر على بيانات خارجية): سلامة البنية: أخضر إن كانت بلا أجزاء متعددة أو مراكز غير محددة. عدم اليقين: أخضر 75% فأكثر، أصفر 50 إلى 74، أحمر أقل. دعم النطاق: أخضر تشابه 0.55 فأكثر، أصفر 0.35 إلى 0.54، أحمر أقل. موثوقية النموذج: أخضر إن كان 11 تنبؤاً أو أكثر بمستوى «أثق به»، أصفر 7 إلى 10، أحمر أقل. قابلية التصنيع: أخضر SA حتى 3.5، أصفر حتى 6، أحمر أعلى. الجدة والأدلة التجريبية للعلم فقط بلا حكم.",
 "en": "Colour criteria (judgment-based, not calibrated on external data): structure: green if single fragment and no undefined stereocentres. Uncertainty: green 75%+, yellow 50-74, red below. Domain: green similarity 0.55+, yellow 0.35-0.54, red below. Model reliability: green if 11+ predictions rated trust, yellow 7-10, red below. Synthesizability: green SA up to 3.5, yellow up to 6, red above. Novelty and experimental evidence are informational only."}
LBL_AR = {"Heavy atoms": "الذرات الثقيلة", "Lipinski violations": "مخالفات ليبنسكي", "Rings": "الحلقات", "Formula": "الصيغة", "app_version": "نسخة التطبيق", "model_sha256": "بصمة النموذج", "rdkit": "نسخة RDKit", "input_sha256": "بصمة المدخل", "generated_utc": "وقت التوليد (UTC)", "note": "ملاحظة"}
ICON = {"green": "🟢", "yellow": "🟡", "red": "🔴", "info": "🔵", "none": "⚪"}


VS = {"ar": {"Verified": "متحقق حسابياً", "Pending": "قيد التحقق", "Unknown": "غير معروف"}, "en": {"Verified": "Verified (computational)", "Pending": "Pending", "Unknown": "Unknown"}}


SRC = {"smiles": ("إدخال SMILES نصي", "Typed SMILES"), "name": ("بحث باسم المركب في PubChem", "PubChem name lookup"), "draw": ("رسم في المحرر", "Structure editor drawing"), "batch": ("ملف دفعة CSV أو نص ملصق", "Batch CSV or pasted text")}


def build(r, model, pc=None, lang="ar", src="smiles"):
    ar = lang == "ar"
    smi = r["smiles"]
    m = Chem.MolFromSmiles(smi)
    frags = len(Chem.GetMolFrags(m))
    centers = Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)
    undefined = sum(1 for _, c in centers if c == "?")
    radicals = sum(a.GetNumRadicalElectrons() for a in m.GetAtoms())
    ad = r["ad"]
    tt = v6.trust_table(r, ad, model.test_metrics, lang)
    n_high = sum(1 for x in tt if x[1] == "high")
    near = ultra.similar(smi, 3)
    top_name, top_sim = near[0]
    axes = []

    def ax(name_ar, name_en, level, why_ar, why_en):
        axes.append({"axis": name_ar if ar else name_en, "level": level, "why": why_ar if ar else why_en})

    # 1 structure integrity
    probs_ar, probs_en = [], []
    if frags > 1:
        probs_ar.append(f"{frags} أجزاء (ملح أو خليط)"); probs_en.append(f"{frags} fragments (salt or mixture)")
    if undefined:
        probs_ar.append(f"{undefined} مركز كيرالي غير محدد"); probs_en.append(f"{undefined} undefined stereocentre(s)")
    if radicals:
        probs_ar.append("إلكترونات حرة"); probs_en.append("radical electrons")
    ax("سلامة البنية", "Structure integrity", "green" if not probs_ar else "yellow",
       "البنية صالحة وتجاوزت فحص RDKit، جزء واحد بلا مشاكل ظاهرة." if not probs_ar else "البنية صالحة لكن: " + "، ".join(probs_ar),
       "Valid structure, passes RDKit sanitization, single fragment, no visible issues." if not probs_en else "Valid structure but: " + ", ".join(probs_en))
    # 2 uncertainty
    c = r["confidence"]
    ax("عدم اليقين", "Uncertainty", "green" if c >= 75 else "yellow" if c >= 50 else "red",
       f"درجة الثبات {c}% عبر 30 تمريرة عشوائية.", f"Stability {c}% across 30 stochastic passes.")
    # 3 domain
    ax("دعم النطاق", "Domain support", {"in": "green", "edge": "yellow", "out": "red"}[ad["band"]],
       f"أعلى تشابه مع بيانات التدريب {ad['max_sim']} ({v6.AD_TXT[ad['band']][1]})",
       f"Highest similarity to training data {ad['max_sim']} ({v6.AD_TXT[ad['band']][2]})")
    # 4 model reliability on these endpoints
    ax("موثوقية النموذج لهذا الجزيء", "Model reliability here", "green" if n_high >= 11 else "yellow" if n_high >= 7 else "red",
       f"{n_high} من {len(tt)} تنبؤاً بمستوى «أثق به» في جدول الثقة.", f"{n_high} of {len(tt)} predictions rated 'trust' in the trust table.")
    # 5 synthesizability
    s = r["sa"]
    ax("قابلية التصنيع", "Synthesizability", "none" if s is None else "green" if s <= 3.5 else "yellow" if s <= 6 else "red",
       f"درجة SA {s}: {v6.sa_text(s, True)} (تقدير حاسوبي).", f"SA score {s}: {v6.sa_text(s, False)} (computational estimate).")
    # 6 novelty (informational)
    if pc and pc.get("status") == "found":
        pc_ar, pc_en = f" موجود في PubChem (CID {pc['cid']}).", f" Found in PubChem (CID {pc['cid']})."
    elif pc and pc.get("status") == "not_found":
        pc_ar, pc_en = " غير موجود في PubChem بهذه البنية.", " Not found in PubChem with this structure."
    elif pc:
        pc_ar, pc_en = " تعذّر الاتصال بـ PubChem.", " PubChem was unreachable."
    else:
        pc_ar, pc_en = " لم يُفحص في PubChem.", " PubChem not checked."
    kind_ar = "مطابق لدواء معروف في مكتبتنا" if top_sim >= 0.95 else "قريب من دواء معروف" if top_sim >= 0.55 else "بعيد عن مكتبة الأدوية لدينا"
    kind_en = "matches a known drug in our library" if top_sim >= 0.95 else "close to a known drug" if top_sim >= 0.55 else "far from our drug library"
    ax("الجدة", "Novelty", "info", f"{kind_ar} (الأقرب: {top_name} {top_sim*100:.0f}%). أعلى تشابه مع {ad['n_ref']:,} جزيء تدريب: {ad['max_sim']}." + pc_ar,
       f"{kind_en} (nearest: {top_name} {top_sim*100:.0f}%). Highest similarity to {ad['n_ref']:,} training molecules: {ad['max_sim']}." + pc_en)
    # 7 experimental evidence (always none)
    ax("الأدلة التجريبية", "Experimental evidence", "none", "لا توجد أدلة تجريبية في هذا الملف.", "No experimental evidence is included in this file.")

    R = r["rules"]
    props = [("MW", round(R["mw"], 2)), ("LogP", round(R["logp"], 2)), ("TPSA", round(R["tpsa"], 1)), ("HBD", R["hbd"]), ("HBA", R["hba"]),
             ("RotB", R["rot"]), ("Rings", R["rings"]), ("Heavy atoms", R["heavy"]), ("QED", round(R["qed"], 3)),
             ("Lipinski violations", R["lipinski_viol"]), ("SA", s), ("CNS-MPO (4/6)", r["mpo"])]
    preds = []
    for t, v in r["tasks"].items():
        val = f"{v['value']:.2f} ± {v['uncertainty']:.2f}" if v["type"] == "regression" else f"{v['prob']*100:.0f}% ± {v['uncertainty']*100:.0f}"
        lvl = next(x[1] for x in tt if x[0] == t)
        preds.append({"endpoint": t, "value": val, "trust": v6.TRUST_NAME[lang][lvl], "test_metric": model.test_metrics.get(t, {})})
    validation = [
        {"item": "صلاحية البنية" if ar else "Structure validity", "status": "Verified",
         "note": "تحقق حسابي بـ RDKit (ليس تجريبياً)." if ar else "Computational check with RDKit (not experimental)."},
        {"item": "الخصائص المحسوبة" if ar else "Computed properties", "status": "Verified",
         "note": "حسابات دقيقة، تحقق حسابي فقط." if ar else "Exact calculations, computational verification only."},
        {"item": "تنبؤات الذكاء الاصطناعي" if ar else "AI predictions", "status": "Pending",
         "note": "تقديرية، لم تُقارَن بقياس مخبري." if ar else "Estimates, not compared with lab measurements."},
        {"item": "التحقق التجريبي" if ar else "Experimental validation", "status": "Unknown",
         "note": "لا بيانات تجريبية متوفرة هنا." if ar else "No experimental data available here."},
    ]
    now = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    core = {"smiles": smi, "predictions": [(p["endpoint"], p["value"]) for p in preds], "model_sha256": model_sha()}
    d = {
        "report_id": "MS-" + hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:12].upper(),
        "identity": {"canonical_smiles": smi, "inchikey": Chem.MolToInchiKey(m), "formula": rdMolDescriptors.CalcMolFormula(m),
                     "source": SRC.get(src, SRC["smiles"])[0 if ar else 1],
                     "pubchem": pc or {"status": "not_checked"}, "fragments": frags},
        "properties": {"values": props, "tag": TAG[lang][CALC]},
        "prediction": {"rows": preds, "tag": TAG[lang][PRED], "model": "MolSolveAI GNN+Morgan fingerprint", "passes": 30, "seed": 0,
                       "domain": ad},
        "novelty": {"nearest_known_drugs": near},
        "axes": axes,
        "validation": validation,
        "reproducibility": {"app_version": APP_VERSION, "model_sha256": model_sha(), "rdkit": rdBase.rdkitVersion,
                            "input_sha256": hashlib.sha256(smi.encode()).hexdigest(), "generated_utc": now,
                            "note": "نفس المدخل ونفس نسخة النموذج يعطيان النتائج نفسها (بذرة ثابتة)." if ar else "Same input and model version give the same results (fixed seed)."},
        "disclaimer": config.DISCLAIMER,
        "scope_of_validity": (SCOPE_AR if ar else SCOPE_EN),
    }
    return d


def pc_cell(p, ar):
    st_ = p.get("status")
    if st_ == "found":
        return (("موجود، CID %s " if ar else "Found, CID %s ") % p["cid"]) + "<a href='%s'>%s</a>" % (html.escape(p["url"]), "PubChem")
    return {"not_found": "غير موجود بهذه البنية" if ar else "Not found with this structure",
            "unavailable": "تعذّر الاتصال بـ PubChem" if ar else "PubChem unreachable"}.get(st_, "لم يُفحص" if ar else "Not checked")


def html_report(d, lang="ar"):
    ar = lang == "ar"
    e = html.escape
    css = ("<style>body{font-family:Tahoma,Arial,sans-serif;max-width:820px;margin:auto;padding:20px;direction:%s}"
           "h1{color:#0b5cad;border-bottom:4px solid #f5b301}h2{color:#0b5cad;margin-top:22px}table{border-collapse:collapse;width:100%%}"
           "td,th{border:1px solid #ddd;padding:6px;text-align:start}.tag{background:#f3f8ff;border-radius:6px;padding:2px 8px;font-size:.85em}"
           "@media print{.noprint{display:none}}</style>") % ("rtl" if ar else "ltr")
    T = lambda a, b: a if ar else b
    i = d["identity"]
    out = [f"<html><head><meta charset='utf-8'>{css}</head><body><h1>MolSolveAI | منارة الإبداع</h1>",
           f"<p><b>{T('ملف الأدلة','Evidence Dossier')}</b> {e(d['report_id'])}</p>",
           f"<h2>1. {T('الهوية','Identity')}</h2><table>",
           f"<tr><td>SMILES</td><td><code>{e(i['canonical_smiles'])}</code></td></tr>",
           f"<tr><td>InChIKey</td><td>{e(i['inchikey'])}</td></tr><tr><td>{T('الصيغة','Formula')}</td><td>{e(i['formula'])}</td></tr>",
           f"<tr><td>{T('المصدر','Source')}</td><td>{e(i['source'])}</td></tr>",
           f"<tr><td>PubChem</td><td>{pc_cell(i['pubchem'], ar)}</td></tr></table>",
           f"<h2>2. {T('الخصائص','Properties')} <span class='tag'>{e(d['properties']['tag'])}</span></h2><table>"]
    out += [f"<tr><td>{e(LBL_AR.get(str(k), str(k)) if ar else str(k))}</td><td dir='ltr'>{e(str(v))}</td><td><span class='tag'>{e(d['properties']['tag'])}</span></td></tr>" for k, v in d["properties"]["values"]]
    out.append(f"</table><h2>3. {T('تنبؤ الذكاء الاصطناعي','AI prediction')} <span class='tag'>{e(d['prediction']['tag'])}</span></h2><table>")
    out += [f"<tr><td>{e(p['endpoint'])}</td><td dir='ltr'>{e(p['value'])}</td><td>{e(p['trust'])}</td><td><span class='tag'>{e(d['prediction']['tag'])}</span></td></tr>" for p in d["prediction"]["rows"]]
    out.append(f"</table><h2>4. {T('لوحة المحاور (بلا درجة واحدة)','Axis panel (no single score)')}</h2><table>")
    out += [f"<tr><td><span style='color:{COL[a['level']]}'>&#9679;</span> {e(a['axis'])}</td><td>{e(a['why'])}</td></tr>" for a in d["axes"]]
    out.append(f"</table><h2>5. {T('حالة التحقق','Verification status')}</h2><table>")
    out += [f"<tr><td>{e(v['item'])}</td><td><b>{e(VS[lang][v['status']])}</b></td><td>{e(v['note'])}</td></tr>" for v in d["validation"]]
    out.append(f"</table><h2>6. {T('سجل إعادة الإنتاج','Reproducibility record')}</h2><table>")
    out += [f"<tr><td>{e(LBL_AR.get(k, k) if ar else k)}</td><td dir='ltr'>{e(str(v))}</td></tr>" for k, v in d["reproducibility"].items()]
    out.append("</table><p><small>" + e(CRITERIA[lang]) + "</small></p><table>")
    out.append(f"</table><p><b>{e(d['scope_of_validity'])}</b></p><p><small>{e(d['disclaimer'])}</small></p></body></html>")
    return "".join(out)
