"""V6: applicability domain, honest trust table, SA score, CNS-MPO (approx), rule-based molecule detective,
share card, teacher worksheet, model card. Everything here is exact calculation or transparent rules (no LLM)."""
import base64, html, io
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

_HERE = Path(__file__).resolve().parent
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
_REF = None


def _ref():
    global _REF
    if _REF is None:
        _REF = np.load(_HERE / "train_fps.npz")["fps"]
    return _REF


# ---------- 1. applicability domain ----------
def domain(smiles):
    m = Chem.MolFromSmiles(smiles)
    fp = np.packbits(np.array(list(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)), dtype=np.uint8))
    R = _ref()
    inter = _POP[np.bitwise_and(R, fp)].sum(1)
    union = _POP[R].sum(1) + _POP[fp].sum() - inter
    sim = inter / np.maximum(union, 1)
    top = np.sort(sim)[-3:]
    mx = float(top[-1])
    band = "in" if mx >= 0.55 else "edge" if mx >= 0.35 else "out"
    return {"max_sim": round(mx, 2), "mean_top3": round(float(top.mean()), 2), "band": band, "n_ref": int(R.shape[0])}


AD_TXT = {
    "in": ("🟢", "داخل نطاق النموذج: يوجد جزيئات مشابهة له في بيانات التدريب.", "Inside the model's domain: similar molecules exist in the training data."),
    "edge": ("🟡", "على حافة النطاق: تشابه متوسط مع بيانات التدريب، تعامل مع التنبؤ بحذر.", "At the edge of the domain: only moderate similarity to training data, treat predictions with care."),
    "out": ("🔴", "خارج النطاق: لا يشبه بيانات التدريب، التنبؤات هنا أقرب للتخمين.", "Outside the domain: unlike the training data, predictions here are close to guesses."),
}


# ---------- 2. SA score and CNS-MPO ----------
def sa(smiles):
    try:
        import sascorer
        return round(float(sascorer.calculateScore(Chem.MolFromSmiles(smiles))), 1)
    except Exception:
        return None


def sa_text(v, ar=True):
    if v is None:
        return "-"
    if v <= 3.5:
        return "سهل التصنيع نسبياً" if ar else "relatively easy to make"
    if v <= 6:
        return "تصنيع متوسط الصعوبة" if ar else "moderate to make"
    return "صعب التصنيع" if ar else "hard to make"


def _lin(x, a, b):  # 1 at a, 0 at b
    return float(min(1, max(0, (b - x) / (b - a))))


def cns_mpo(R):
    """Wager-style CNS MPO using 4 of the 6 published criteria (no pKa/logD): score 0-4."""
    s = _lin(R["logp"], 3, 5) + _lin(R["mw"], 360, 500) + _lin(R["hbd"], 0.5, 3.5)
    t = R["tpsa"]
    s += 0 if t <= 20 else (t - 20) / 20 if t < 40 else 1 if t <= 90 else _lin(t, 90, 120)
    return round(s, 1)


# ---------- 3. trust table ----------
def trust_table(r, ad, metrics, lang="ar"):
    ar = lang == "ar"
    rows = []
    for t, v in r["tasks"].items():
        unc = v["uncertainty"]
        auc = (metrics.get(t) or {}).get("AUC")
        pts, why = 0, []
        if ad["band"] == "in":
            pts += 2
        elif ad["band"] == "edge":
            pts += 1
            why.append("حافة النطاق" if ar else "edge of domain")
        else:
            why.append("خارج النطاق" if ar else "outside domain")
        if v["type"] == "classification":
            decisive = abs(v["prob"] - 0.5) >= 0.2
            pts += (1 if unc < 0.12 else 0) + (1 if decisive else 0)
            if not decisive:
                why.append("قريب من 50% وغير حاسم" if ar else "near 50%, not decisive")
            if unc >= 0.12:
                why.append("تذبذب بين التمريرات" if ar else "unstable across passes")
            if auc is not None and auc < 0.72:
                pts -= 1
                why.append(f"النموذج ضعيف لهذا الاختبار (AUC {auc:.2f})" if ar else f"model weak on this assay (AUC {auc:.2f})")
        else:
            pts += 2 if unc < 0.35 else 0
            if unc >= 0.35:
                why.append("تذبذب مرتفع" if ar else "high spread")
        lvl = "high" if pts >= 4 else "mid" if pts >= 2 else "low"
        rows.append((t, lvl, "؛ ".join(why) if why else ("كل المؤشرات جيدة" if ar else "all signals good")))
    return rows


TRUST_ICON = {"high": "🟢", "mid": "🟡", "low": "🔴"}
TRUST_NAME = {"ar": {"high": "أثق به", "mid": "بحذر", "low": "لا أثق به"}, "en": {"high": "trust", "mid": "careful", "low": "don't trust"}}


# ---------- 4. molecule detective (transparent rules, not an LLM) ----------
def detective(r, ad, sav, mpo, lang="ar"):
    ar = lang == "ar"
    R, ts = r["rules"], r["tasks"]
    probs, fixes = [], []
    if ts["LogS"]["value"] < -4.5:
        if R["logp"] > 3.5:
            probs.append(("ذوبانيته ضعيفة، والسبب الأرجح دهنيته العالية (LogP %.1f): الجزيء يفضّل الدهون على الماء." % R["logp"]) if ar
                         else "Poor solubility, most likely from high lipophilicity (LogP %.1f): it prefers fat over water." % R["logp"])
            fixes.append("أضف مجموعة قطبية (OH أو NH2 أو N في الحلقة) أو أزل حلقة عطرية." if ar else "Add a polar group (OH, NH2, ring N) or remove an aromatic ring.")
        else:
            probs.append("ذوبانيته ضعيفة رغم أن LogP معتدل، قد يكون السبب كبر الحجم أو تماسك البلورة." if ar else "Poor solubility even with moderate LogP; size or crystal packing may be the cause.")
            fixes.append("جرّب كسر التناظر أو إضافة مجموعة قابلة للتأين." if ar else "Try breaking symmetry or adding an ionizable group.")
    if R["tpsa"] > 140 or R["hbd"] > 5:
        probs.append(("قطبيته عالية (TPSA %.0f)، وهذا يصعّب عبور الأغشية والامتصاص الفموي." % R["tpsa"]) if ar
                     else "High polarity (TPSA %.0f) makes membrane crossing and oral absorption hard." % R["tpsa"])
        fixes.append("قلّل مانحات الهيدروجين أو غطِّ مجموعات OH/NH (مثلاً إلى OMe)." if ar else "Reduce H-bond donors or cap OH/NH groups (e.g. to OMe).")
    if R["mw"] > 500:
        probs.append(("وزنه الجزيئي كبير (%.0f) ويخالف قاعدة ليبنسكي." % R["mw"]) if ar else "Large molecular weight (%.0f), breaks Lipinski." % R["mw"])
        fixes.append("بسّط البنية وأزل الأجزاء غير الضرورية." if ar else "Simplify the structure and drop unnecessary parts.")
    if r["tox_flags"]:
        probs.append(("إشارات سمّية في %d اختبار: %s." % (len(r["tox_flags"]), "، ".join(r["tox_flags"]))) if ar
                     else "Toxicity signals in %d assays: %s." % (len(r["tox_flags"]), ", ".join(r["tox_flags"])))
        fixes.append("افحص الخريطة الذرية لمعرفة الذرات المسؤولة وعدّلها." if ar else "Check the atom map to find responsible atoms and edit them.")
    if r["alerts"]:
        probs.append(("فيه مجموعات تُعرف بالتنبيهات البنيوية: %s." % "، ".join(r["alerts"][:3])) if ar else "Contains known structural alerts: %s." % ", ".join(r["alerts"][:3]))
    if r["alerts"]:
        fixes.append("راجع المجموعة المنبّهة: قد تكون مقبولة في سياقك أو تحتاج استبدالاً بمجموعة أكثر ثباتاً." if ar
                     else "Review the flagged group: it may be acceptable in your context or need swapping for a more stable one.")
    if sav is not None and sav > 6:
        probs.append(("تصنيعه صعب (SA %.1f) فأي تحسين مقترح عليه قد لا يكون عملياً." % sav) if ar else "Hard to synthesize (SA %.1f); improvements may be impractical." % sav)
    good = []
    if R["lipinski_viol"] == 0:
        good.append("يطابق قاعدة ليبنسكي" if ar else "passes Lipinski")
    if R["qed"] >= 0.6:
        good.append(("شبه الدواء مرتفع (QED %.2f)" % R["qed"]) if ar else "high drug-likeness (QED %.2f)" % R["qed"])
    if mpo >= 3.5:
        good.append(("مناسب للجهاز العصبي المركزي (CNS-MPO %.1f)" % mpo) if ar else "good for the CNS (CNS-MPO %.1f)" % mpo)
    head = ("🕵️ ملف الجزيء: وزن %.0f، LogP %.1f، %d حلقات." % (R["mw"], R["logp"], R["rings"])) if ar else \
        "🕵️ Molecule file: MW %.0f, LogP %.1f, %d rings." % (R["mw"], R["logp"], R["rings"])
    out = [head]
    if good:
        out.append(("✅ نقاط القوة: " + "، ".join(good) + ".") if ar else "✅ Strengths: " + ", ".join(good) + ".")
    else:
        out.append("لا نقاط قوة بارزة." if ar else "No standout strengths.")
    if probs:
        out.append(("⚠️ المشاكل:\n\n" if ar else "⚠️ Problems:\n\n") + "\n\n".join("- " + p for p in probs))
        if fixes:
            out.append(("🔧 خطوات مقترحة:\n\n" if ar else "🔧 Suggested steps:\n\n") + "\n\n".join("%d) %s" % (i + 1, f) for i, f in enumerate(dict.fromkeys(fixes))))
    else:
        out.append("✅ لا مشاكل كبيرة ظاهرة." if ar else "✅ No major problems visible.")
    out.append(("🔎 موثوقية التحليل: " if ar else "🔎 Reliability: ") + AD_TXT[ad["band"]][1 if ar else 2])
    out.append("هذا شرح مبني على قواعد شفافة وحسابات دقيقة، وليس نموذجاً لغوياً ولا نصيحة علاجية." if ar else
               "This explanation comes from transparent rules and exact calculations, not an LLM, and is not medical advice.")
    return "\n\n".join(out)


# ---------- 5. share card ----------
def _ar(s):
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(s))
    except Exception:
        return s


def share_card(r, sav, mpo, ov):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    d = rdMolDraw2D.MolDraw2DCairo(560, 420)
    d.DrawMolecule(Chem.MolFromSmiles(r["smiles"])); d.FinishDrawing()
    img = np.array(Image.open(io.BytesIO(d.GetDrawingText())).convert("RGB"))
    fig = plt.figure(figsize=(7.2, 9), dpi=140, facecolor="#0b5cad")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0.04, 0.27), 0.92, 0.55, color="white", zorder=1))
    ax2 = fig.add_axes([0.1, 0.3, 0.8, 0.5]); ax2.imshow(img); ax2.axis("off")
    ax.text(0.5, 0.925, "MolSolveAI", color="white", ha="center", fontsize=32, fontweight="bold")
    from matplotlib import font_manager as fm
    fp = fm.FontProperties(fname=str(_HERE / "NotoSansArabic-Bold.ttf")) if (_HERE / "NotoSansArabic-Bold.ttf").exists() else None
    if fp is not None:
        ax.text(0.5, 0.885, _ar("منارة الإبداع"), color="#f5b301", ha="center", fontsize=21, fontproperties=fp)
    else:
        ax.text(0.5, 0.88, "Manarat Al-Ibdaa", color="#f5b301", ha="center", fontsize=17, fontweight="bold")
    col = {"green": "#f5b301", "yellow": "#f5b301", "red": "#ff8a80"}[ov]
    lab = {"green": "Promising", "yellow": "Mixed signals", "red": "Risky"}[ov]
    ax.text(0.5, 0.845, f"Overall: {lab}", color=col, ha="center", fontsize=17, fontweight="bold")
    R, ts = r["rules"], r["tasks"]
    lines = [f"QED {R['qed']:.2f}  |  MW {R['mw']:.0f}  |  LogP {R['logp']:.1f}",
             f"Solubility LogS {ts['LogS']['value']:.1f}  |  Brain {ts['BBB']['prob']*100:.0f}%",
             f"Tox signals {len(r['tox_flags'])}/12  |  SA {sav if sav is not None else '-'}  |  CNS-MPO {mpo}/4"]
    for i, l in enumerate(lines):
        ax.text(0.5, 0.205 - i * 0.043, l, color="white", ha="center", fontsize=13)
    ax.text(0.5, 0.058, "Manarat Al-Ibdaa  |  MolSolveAI", color="#f5b301", ha="center", fontsize=11, fontweight="bold")
    ax.text(0.5, 0.022, "AI estimate for learning only, not medical advice", color="#cfe3f7", ha="center", fontsize=9)
    b = io.BytesIO(); fig.savefig(b, format="png", facecolor=fig.get_facecolor()); plt.close(fig)
    return b.getvalue()


# ---------- 6. teacher worksheet ----------
def _png_b64(smiles, size=(360, 280)):
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    d.DrawMolecule(Chem.MolFromSmiles(smiles)); d.FinishDrawing()
    return base64.b64encode(d.GetDrawingText()).decode()


def teacher_sheet(items, lang="ar", school=""):
    """items: list of (name, smiles, rules dict). Returns printable HTML (worksheet + answer key)."""
    ar = lang == "ar"
    css = ("<style>body{font-family:Tahoma,Arial,sans-serif;max-width:780px;margin:auto;padding:20px;direction:%s}"
           "h1{color:#0b5cad;border-bottom:4px solid #f5b301}.q{margin:10px 0;padding:8px;border:1px solid #ddd;border-radius:8px}"
           ".page{page-break-after:always}.k{background:#f3f8ff}.line{border-bottom:1px dotted #999;height:22px}"
           "@media print{.noprint{display:none}}</style>") % ("rtl" if ar else "ltr")
    head = ("<h1>MolSolveAI | منارة الإبداع</h1>" + (f"<p>{html.escape(school)}</p>" if school else "") +
            ("<p>الاسم: ................... التاريخ: ............</p>" if ar else "<p>Name: ................... Date: ............</p>"))
    ws, key = [], []
    for n, (name, smi, R) in enumerate(items, 1):
        yes, no = ("نعم", "لا") if ar else ("yes", "no")
        qs = [("كم عدد الذرات الثقيلة (غير الهيدروجين) في هذا الجزيء؟" if ar else "How many heavy (non-hydrogen) atoms does this molecule have?", str(R["heavy"])),
              ("كم عدد الحلقات؟" if ar else "How many rings does it contain?", str(R["rings"])),
              ("هل الوزن الجزيئي أقل من 500؟" if ar else "Is the molecular weight below 500?", f"{yes if R['mw'] < 500 else no} (MW = {R['mw']:.0f})"),
              ("هل يطابق قاعدة ليبنسكي الخمسية؟ اشرح." if ar else "Does it follow Lipinski's rule of five? Explain.",
               (f"المخالفات = {R['lipinski_viol']}: MW {R['mw']:.0f}، LogP {R['logp']:.1f}، HBD {R['hbd']}، HBA {R['hba']}." if ar
                else f"Violations = {R['lipinski_viol']}: MW {R['mw']:.0f}, LogP {R['logp']:.1f}, HBD {R['hbd']}, HBA {R['hba']}.")),
              ("هل تتوقع ذوبانية جيدة في الماء؟ علّل بالدهنية." if ar else "Do you expect good water solubility? Reason with lipophilicity.",
               (f"LogP = {R['logp']:.1f}؛ كلما ارتفع LogP قلّت الذوبانية عادة." if ar else f"LogP = {R['logp']:.1f}; higher LogP usually means lower solubility."))]
        blk = f"<div class='q'><b>{'الجزيء' if ar else 'Molecule'} {n}: {html.escape(name)}</b><br><img src='data:image/png;base64,{_png_b64(smi)}'>"
        kblk = blk
        for i, (q, a) in enumerate(qs, 1):
            blk += f"<p>{i}) {q}</p><div class='line'></div>"
            kblk += f"<p>{i}) {q}<br><b>✔ {html.escape(a)}</b></p>"
        ws.append(blk + "</div>"); key.append(kblk + "</div>")
    foot = ("<p class='noprint'>للطباعة كـ PDF: افتح الملف في المتصفح ثم اطبع واختر حفظ كـ PDF.</p>" if ar
            else "<p class='noprint'>To get a PDF: open in a browser, print, choose Save as PDF.</p>")
    return (f"<html><head><meta charset='utf-8'>{css}</head><body>{head}<h2>{'ورقة العمل' if ar else 'Worksheet'}</h2>{''.join(ws)}{foot}"
            f"<div class='page'></div><h1>{'مفتاح الإجابة (للمعلم)' if ar else 'Answer key (teacher)'}</h1><div class='k'>{''.join(key)}</div>"
            f"<p><small>{'الحسابات من RDKit بدقة وليست تخميناً.' if ar else 'Values computed exactly with RDKit, not AI guesses.'}</small></p></body></html>")


# ---------- 7. transparent check on well-known drugs ----------
KNOWN_BBB = {"Caffeine": True, "Diazepam": True, "Morphine": True, "Nicotine": True, "Fluoxetine": True, "Sertraline": True,
             "Lidocaine": True, "Dopamine": False, "Atenolol": False, "Penicillin G": False, "Amoxicillin": False,
             "Metformin": False, "Acyclovir": False, "Ranitidine": False, "Cetirizine": False}


def check_known(analyze, lib):
    rows, hit = [], 0
    for n, truth in KNOWN_BBB.items():
        if n not in lib:
            continue
        p = analyze(lib[n], 10)["tasks"]["BBB"]["prob"]
        ok = (p > 0.5) == truth
        hit += ok
        rows.append({"drug": n, "expected_crosses_brain": truth, "model_prob_%": round(p * 100), "match": "✅" if ok else "❌",
                     "max_similarity_to_training": domain(lib[n])["max_sim"]})
    return rows, hit


MODEL_CARD = {
 "ar": """**البنية:** شبكة عصبية رسومية (GNN) مع بصمة Morgan، استدلال بـ numpy.

**البيانات:** ESOL (الذوبانية)، BBBP (عبور الدماغ)، Tox21 (12 اختبار سمّية)، كلها عامة ومفتوحة.

**التقييم:** تقسيم بالسقالة الكيميائية (أصعب من العشوائي): LogS R² 0.71، BBB AUC 0.96، Tox21 متوسط AUC 0.77 (الأضعف: Aromatase 0.65، SR-ARE 0.66).

**نطاق التطبيق:** تشابه Tanimoto مع نحو 9,500 جزيء تدريب فريد: داخل ≥ 0.55، حافة 0.35 إلى 0.55، خارج أقل من 0.35 (حدود عملية وضعناها، تقريبية).

**ما لا يفعله:** لا استقلاب ولا CYP ولا hERG ولا حرائك دوائية، ولا يضمن قابلية التصنيع، وليس أداة قرار علاجي.

**SA:** درجة Ertl وSchuffenhauer المضمّنة في RDKit. **CNS-MPO:** نسخة مبسطة بـ 4 من 6 معايير (بدون pKa وlogD).""",
 "en": """**Architecture:** graph neural network plus Morgan fingerprint, numpy inference.

**Data:** ESOL (solubility), BBBP (brain penetration), Tox21 (12 toxicity assays), all public.

**Evaluation:** scaffold split (harder than random): LogS R² 0.71, BBB AUC 0.96, Tox21 mean AUC 0.77 (weakest: Aromatase 0.65, SR-ARE 0.66).

**Applicability domain:** Tanimoto similarity to ~9,500 unique training molecules: inside >= 0.55, edge 0.35 to 0.55, outside < 0.35 (our practical, approximate cut-offs).

**What it does not do:** no metabolism, CYP, hERG or pharmacokinetics, no synthesis guarantee, not for clinical decisions.

**SA:** Ertl & Schuffenhauer score shipped with RDKit. **CNS-MPO:** simplified, 4 of 6 criteria (no pKa/logD).""",
}
