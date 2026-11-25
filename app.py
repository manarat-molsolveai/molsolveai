"""MolSolveAI web app. Run: streamlit run app.py"""
import io, json, os
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import config, infer, druglike, ultra, ultra2, v6, v7, dossier
from chem import canonical
from i18n import T, PLANS

st.set_page_config(page_title="MolSolveAI", page_icon="🧪", layout="centered")

if "lang" not in st.session_state:
    st.session_state.lang = "ar"
st.session_state.setdefault("used", 0)
st.session_state.setdefault("plan", "free")

lang = st.sidebar.radio("Language / اللغة", ["ar", "en"], format_func=lambda x: "العربية" if x == "ar" else "English",
                        index=0 if st.session_state.lang == "ar" else 1)
st.session_state.lang = lang
t = T[lang]
if lang == "ar":
    st.markdown("<style>.main, .stMarkdown, label, .stTextInput, .stTabs {direction: rtl; text-align: right}</style>",
                unsafe_allow_html=True)

# Optional license keys, from env / Streamlit secrets. Empty by default: nothing unlocks without the owner.
def _keys():
    raw = os.getenv("MOLSOLVE_PRO_KEYS", "")
    try:
        raw = raw or st.secrets.get("MOLSOLVE_PRO_KEYS", "")
    except Exception:
        pass
    return {k.strip() for k in str(raw).split(",") if k.strip()}

with st.sidebar:
    key = st.text_input(t["key"], type="password")
    if key:
        if key in _keys():
            st.session_state.plan = "pro"; st.success(t["key_ok"])
        else:
            st.error(t["key_bad"])
    plan = PLANS[st.session_state.plan]
    st.write(f"{t['plan']}: **{plan['name']}**")
    st.caption(t["left"].format(n=max(0, plan["single"] - st.session_state.used)))

LOGO = ("<svg width='46' height='46' viewBox='0 0 64 64' xmlns='http://www.w3.org/2000/svg'><polygon points='24,58 40,58 36,22 28,22' fill='#0b5cad'/>"
        "<rect x='26' y='14' width='12' height='9' fill='#f5b301'/><polygon points='24,14 32,6 40,14' fill='#0b5cad'/>"
        "<polygon points='4,12 26,17 26,19 4,22' fill='#f5b301' opacity='.55'/><polygon points='60,12 38,17 38,19 60,22' fill='#f5b301' opacity='.55'/>"
        "<rect x='20' y='56' width='24' height='4' fill='#0a3d75'/></svg>")
st.markdown(f"<div style='display:flex;align-items:center;gap:12px;direction:ltr'>{LOGO}<div><div style='font-size:2rem;font-weight:700'>MolSolveAI</div>"
            f"<div style='color:#0b5cad;font-weight:600'>{'منارة الإبداع  |  Manarat Al-Ibdaa'}</div></div></div>", unsafe_allow_html=True)
st.caption(t["tag"])
st.warning(t["disc"])

if not infer.NPZ_PATH.exists():
    st.error("Model file missing."); st.stop()


@st.cache_resource
def get():
    return infer.load()


model = get()
EXAMPLES = {"Aspirin": "CC(=O)Oc1ccccc1C(=O)O", "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
            "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Benzene": "c1ccccc1", "Paracetamol": "CC(=O)Nc1ccc(O)cc1"}


def L(en, ar):
    return en if lang == "en" else ar


def analyze(smi, passes=30):
    c = canonical(smi)
    if c is None:
        return None
    r = infer.predict(c, model, passes=passes)
    r["rules"] = druglike.rules(c)
    r["alerts"] = druglike.alerts(c)
    r["confidence"] = druglike.confidence(r)
    r["tox_flags"] = druglike.tox_summary(r, config.TOX21_TASKS)
    r["ad"] = v6.domain(c)
    r["sa"] = v6.sa(c)
    r["mpo"] = v6.cns_mpo(r["rules"])
    return r


tab1, tab2, tab_cmp, tab_opt, tab_teach, tab_quiz, tab_card, tab_ver, tab3, tab4 = st.tabs([t["tab_single"], t["tab_batch"], t["tab_cmp"], L("Optimizer", "محسّن الجزيء"), L("Teacher mode", "وضع المعلم"), L("Challenge", "تحدي الكيمياء"), L("Model card", "بطاقة النموذج"), L("Verify a report", "تحقق من تقرير"), t["tab_plans"], t["tab_about"]])

@st.cache_data(ttl=86400, show_spinner=False)
def pubchem_cached(smi):
    return dossier.pubchem(smi)


def render(r, key):
    R, ts = r["rules"], r["tasks"]
    c0, c1 = st.columns([1, 2])
    c0.image(Draw.MolToImage(Chem.MolFromSmiles(r["smiles"]), size=(300, 240)))
    m1, m2 = c1.columns(2)
    m1.metric(t["conf"], f"{r['confidence']}%", help=t["conf_help"])
    m2.metric("QED", f"{R['qed']:.2f}", help=t["qed"])
    c1.code(r["smiles"])
    ov, items = ultra.verdict(r, lang)
    st.subheader(f"{ultra.DOT[ov]} {t['verdict']}")
    for col, txt in items:
        st.write(f"{ultra.DOT[col]} {txt}")
    ad = r["ad"]; ic, ta, te = v6.AD_TXT[ad["band"]]
    (st.success if ad["band"] == "in" else st.warning if ad["band"] == "edge" else st.error)(
        f"{ic} {L(te, ta)}  ({L('max similarity', 'أعلى تشابه')} {ad['max_sim']})")
    s1, s2, s3 = st.columns(3)
    s1.metric(L("Synthetic accessibility (SA)", "سهولة التصنيع (SA)"), r["sa"] if r["sa"] is not None else "-",
              help=L("1 = very easy, 10 = very hard. Ertl score from RDKit.", "1 سهل جداً، 10 صعب جداً. درجة Ertl من RDKit."))
    s1.caption(v6.sa_text(r["sa"], lang == "ar"))
    s2.metric("CNS-MPO", f"{r['mpo']}/4", help=L("Simplified (4 of 6 criteria). Higher = better for brain drugs.", "نسخة مبسطة (4 من 6 معايير). الأعلى أفضل لأدوية الدماغ."))
    s3.metric(L("Heavy atoms", "الذرات الثقيلة"), R["heavy"])
    with st.expander(L("📁 Evidence Dossier (status, axes, reproducibility)", "📁 ملف الأدلة (الحالة، المحاور، إعادة الإنتاج)"), expanded=True):
        pcx = st.checkbox(L("Check PubChem presence (needs internet)", "افحص وجوده في PubChem (يحتاج إنترنت)"), value=True, key=f"pc_{key}")
        d = dossier.build(r, model, pubchem_cached(r["smiles"]) if pcx else None, lang, st.session_state.get("src", "smiles") if key == "single" else "batch")
        st.caption(f"{L('Report ID', 'رقم التقرير')}: {d['report_id']}  |  InChIKey: {d['identity']['inchikey']}")
        st.markdown(f"**{L('Axes (no single score)', 'المحاور (بلا درجة واحدة)')}: {len(d['axes'])}**")
        for a_ in d["axes"]:
            st.write(f"{dossier.ICON[a_['level']]} **{a_['axis']}**: {a_['why']}")
        st.caption(dossier.CRITERIA[lang])
        st.markdown(f"**{L('Verification status', 'حالة التحقق')}**")
        st.table(pd.DataFrame([{**v, "status": dossier.VS[lang][v["status"]]} for v in d["validation"]]).rename(columns={"item": L("item", "البند"), "status": L("status", "الحالة"), "note": L("note", "ملاحظة")}))
        st.markdown(f"**{L('Computed properties', 'الخصائص')}** ({dossier.TAG[lang]['calc']})")
        st.dataframe(pd.DataFrame(d["properties"]["values"], columns=[L("property", "الخاصية"), L("value", "القيمة")]).astype(str), hide_index=True)
        st.caption(f"{dossier.TAG[lang]['pred']} / {dossier.TAG[lang]['unverified']}")
        st.info(v7.SCOPE[lang])
        st.json(d["reproducibility"], expanded=False)
        e1, e2 = st.columns(2)
        e1.download_button(L("Dossier (HTML)", "الملف (HTML)"), dossier.html_report(d, lang), f"dossier_{d['report_id']}.html", "text/html", key=f"dh_{key}")
        e2.download_button(L("Dossier (JSON)", "الملف (JSON)"), json.dumps(d, ensure_ascii=False, indent=2, default=str), f"dossier_{d['report_id']}.json", key=f"dj_{key}")
    with st.expander(L("🕵️ Molecule Detective (plain explanation)", "🕵️ محقق الجزيء (شرح مبسط)"), expanded=True):
        st.write(v6.detective(r, ad, r["sa"], r["mpo"], lang))
    with st.expander(L("🔎 Trust table: how much to trust each prediction", "🔎 جدول الثقة: كم أثق بكل تنبؤ")):
        tt = v6.trust_table(r, ad, model.test_metrics, lang)
        st.dataframe(pd.DataFrame({L("prediction", "التنبؤ"): [x[0] for x in tt],
                                   L("trust", "الثقة"): [f"{v6.TRUST_ICON[x[1]]} {v6.TRUST_NAME[lang][x[1]]}" for x in tt],
                                   L("why", "السبب"): [x[2] for x in tt]}), hide_index=True, height=520)
    rc1, rc2 = st.columns(2)
    rc1.caption(t["radar"]); rc1.pyplot(ultra.radar([("", r)]))
    rc2.caption(t["similar"])
    for n_, sc in ultra.similar(r["smiles"]):
        rc2.write(f"**{n_}**: {sc*100:.0f}%")
    st.subheader(t["props"])
    st.table(pd.DataFrame({"MW": [round(R["mw"], 2)], "LogP": [round(R["logp"], 2)], "TPSA": [round(R["tpsa"], 1)],
                           "HBD": [R["hbd"]], "HBA": [R["hba"]], "RotB": [R["rot"]], "Rings": [R["rings"]], "Heavy atoms": [R["heavy"]]}))
    st.subheader(t["rules"])
    st.write(f"Lipinski: {R['lipinski_viol']} | Veber: {R['veber_viol']} | Ghose: {R['ghose_viol']} {t['viol']}")
    st.write(f"**{t['alerts']}:** " + (", ".join(r["alerts"]) if r["alerts"] else t["none"]))
    st.subheader(t["ai"])
    st.write(f"**{t['logs']}:** {ts['LogS']['value']:.2f} ± {ts['LogS']['uncertainty']:.2f}")
    b = ts["BBB"]
    st.write(f"**{t['bbb']}:** {t['yes'] if b['prob'] > .5 else t['no']} ({b['prob']*100:.0f}% ± {b['uncertainty']*100:.0f})")
    st.markdown(f"**{t['tox']}**")
    st.dataframe(pd.DataFrame({"assay": config.TOX21_TASKS,
                               "prob %": [round(ts[k]["prob"] * 100) for k in config.TOX21_TASKS],
                               "± %": [round(ts[k]["uncertainty"] * 100) for k in config.TOX21_TASKS]}), hide_index=True, height=460)
    st.bar_chart(pd.Series({k: ts[k]["prob"] for k in config.TOX21_TASKS}))
    st.write(("**" + t["risk"] + ":** " + ", ".join(r["tox_flags"])) if r["tox_flags"] else t["risk0"])
    st.subheader(L("Why? Atom-level explanation", "لماذا؟ تفسير على مستوى الذرات"))
    ep = st.selectbox(L("Explain which prediction", "فسّر أي تنبؤ"), ["LogS", "BBB"] + config.TOX21_TASKS, key=f"ep_{key}")
    sc = ultra2.atom_scores(r["smiles"], model, ep)
    st.image(ultra2.heatmap_png(r["smiles"], sc))
    st.caption(L("Red atoms push the prediction UP, blue atoms push it DOWN. Approximate: it masks each atom in the graph part of the model only.",
                 "الذرات الحمراء ترفع قيمة التنبؤ، والزرقاء تخفضها. تقريبي: يخفي كل ذرة في الجزء الرسومي من النموذج فقط."))
    try:
        st.download_button(L("🖼️ Share card (PNG)", "🖼️ بطاقة مشاركة (صورة)"), v6.share_card(r, r["sa"], r["mpo"], ov), f"molsolveai_card_{key}.png",
                           "image/png", key=f"dlc_{key}")
    except Exception:
        pass
    st.download_button(t["dl_json"], json.dumps(r, ensure_ascii=False, indent=2), f"molsolveai_{key}.json",
                       "application/json", key=f"dl_{key}")
    st.download_button(t["dl_html"], ultra.html_report([r], lang, t, config.TOX21_TASKS), f"molsolveai_{key}.html",
                       "text/html", key=f"dlh_{key}")


def flat(r):
    R, ts = r["rules"], r["tasks"]
    row = {"smiles": r["smiles"], "confidence": r["confidence"], "QED": round(R["qed"], 3), "MW": round(R["mw"], 2),
           "LogP": round(R["logp"], 2), "TPSA": round(R["tpsa"], 1), "HBD": R["hbd"], "HBA": R["hba"], "RotB": R["rot"],
           "Rings": R["rings"], "Lipinski_viol": R["lipinski_viol"], "Veber_viol": R["veber_viol"], "Ghose_viol": R["ghose_viol"],
           "alerts": "; ".join(r["alerts"]), "LogS": round(ts["LogS"]["value"], 2), "LogS_unc": round(ts["LogS"]["uncertainty"], 2),
           "BBB_prob": round(ts["BBB"]["prob"], 3), "tox_flags": ";".join(r["tox_flags"]), "tox_flag_count": len(r["tox_flags"])}
    for k in config.TOX21_TASKS:
        row[k] = round(ts[k]["prob"], 3)
    return row


with tab1:
    mode = st.radio(t["ultra_mode"], [t["m_smiles"], t["m_name"], t["m_draw"]], horizontal=True)
    st.session_state.setdefault("cur", EXAMPLES["Aspirin"])
    if mode == t["m_smiles"]:
        choice = st.selectbox(t["examples"], [t["own"]] + list(EXAMPLES))
        if choice != t["own"]:
            st.session_state.cur = EXAMPLES[choice]
        smi = st.text_input(t["smiles"], st.session_state.cur)
        st.session_state.cur = smi
    elif mode == t["m_name"]:
        nm = st.text_input(t["name_in"], "ibuprofen")
        if st.button(t["name_go"]):
            found, err = ultra.lookup_name(nm)
            if found:
                st.session_state.cur = found
            else:
                st.error(t["name_bad"])
        st.write(f"{t['name_ok']}: `{st.session_state.cur}`")
        smi = st.session_state.cur
    else:
        try:
            from streamlit_ketcher import st_ketcher
            smi = st_ketcher(st.session_state.cur) or st.session_state.cur
            st.session_state.cur = smi
        except Exception:
            st.error("Editor unavailable.")
            smi = st.session_state.cur
    if st.button(t["go"]):
        if st.session_state.used >= plan["single"]:
            st.error(t["limit_hit"].format(n=plan["single"]))
        else:
            r = analyze(smi)
            st.session_state.src = "smiles" if mode == t["m_smiles"] else "name" if mode == t["m_name"] else "draw"
            if r is None:
                st.error(t["bad"])
            else:
                st.session_state.used += 1
                render(r, "single")
                st.warning(t["disc"])

with tab2:
    st.write(t["batch_help"])
    up = st.file_uploader(t["upload"], type=["csv"])
    pasted = st.text_area(t["paste"], height=120)
    if st.button(t["run"]):
        smiles = []
        if up is not None:
            df = pd.read_csv(up)
            col = next((c for c in df.columns if c.lower() == "smiles"), None)
            if col is None:
                st.error(t["batch_help"]); st.stop()
            smiles = [str(x) for x in df[col].dropna()]
        smiles += [x.strip() for x in pasted.splitlines() if x.strip()]
        cap = plan["batch"]
        if len(smiles) > cap:
            st.info(t["batch_cap"].format(n=cap)); smiles = smiles[:cap]
        aud = v7.audit(smiles)
        seen_c, uniq = set(), []
        for x in smiles:
            c_ = canonical(x)
            if c_ is None or c_ not in seen_c:
                seen_c.add(c_); uniq.append(x)
        smiles = uniq
        res, bad = [], []
        prog = st.progress(0.0)
        for i, x in enumerate(smiles):
            r = analyze(x)
            if r is None:
                bad.append(x)
            else:
                res.append(r)
            prog.progress((i + 1) / max(1, len(smiles)))
        st.session_state.batch = {"res": res, "bad": bad, "audit": aud}
    B = st.session_state.get("batch")
    if B and B.get("audit") and B["audit"]["total"] > 0:
        st.subheader(L("🧪 Data audit (before predictions)", "🧪 تدقيق البيانات (قبل التنبؤ)"))
        st.dataframe(pd.DataFrame(v7.audit_rows(B["audit"], lang == "ar")), hide_index=True)
    if B and B["res"]:
        res = B["res"]
        ur = v7.uncertainty_report(res, model, lang)
        st.subheader(L("🎯 Batch uncertainty report", "🎯 تقرير عدم اليقين للدفعة"))
        u1, u2, u3 = st.columns(3)
        u1.metric(L("Predictions below 'trust'", "تنبؤات دون «أثق به»"), f"{ur['below_trust_pct']}%")
        u2.metric(L("Mean stability", "متوسط الثبات"), f"{ur['mean_stability']}%")
        u3.metric(L("Inside / edge / outside domain", "داخل / حافة / خارج النطاق"), L(f"in {ur['domain']['in']} · edge {ur['domain']['edge']} · out {ur['domain']['out']}", f"داخل {ur['domain']['in']} · حافة {ur['domain']['edge']} · خارج {ur['domain']['out']}"))
        st.dataframe(pd.DataFrame(ur["rows"]), hide_index=True)
        st.caption(L("A calibration curve needs held-out predictions that are not stored with the model, so it is not shown. We do not invent one.", "منحنى المعايرة يحتاج تنبؤات على بيانات اختبار غير محفوظة مع النموذج، لذلك لا نعرضه ولا نختلق واحداً."))
        st.info(v7.SCOPE[lang])
        st.subheader(t["summary"])
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(t["rows"], len(res)); k2.metric(t["n_dl"], sum(r["rules"]["lipinski_viol"] == 0 for r in res))
        k3.metric(t["n_alert"], sum(bool(r["alerts"]) for r in res)); k4.metric(t["n_tox"], sum(bool(r["tox_flags"]) for r in res))
        best = max(res, key=lambda r: r["rules"]["qed"])
        st.write(f"{t['best']}: `{best['smiles']}` (QED {best['rules']['qed']:.2f})")
        sort_by = st.selectbox(t["sort"], ["QED", "confidence", "LogS", "BBB_prob", "tox_flag_count"])
        table = pd.DataFrame([flat(r) for r in res]).sort_values(sort_by, ascending=(sort_by == "tox_flag_count")).reset_index(drop=True)
        st.dataframe(table)
        dl = [dossier.build(r_, model, None, lang, "batch") for r_ in res]
        st.download_button(L("Dossier for all (JSON)", "ملف الأدلة للجميع (JSON)"), json.dumps(dl, ensure_ascii=False, indent=2, default=str), "dossier_batch.json", "application/json", key="dl_batch_dos")
        st.download_button(L("Dossier for all (HTML)", "ملف الأدلة للجميع (HTML)"), "<hr style='page-break-after:always'>".join(dossier.html_report(x, lang) for x in dl), "dossier_batch.html", "text/html", key="dl_batch_dosh")
        st.download_button(t["dl_csv"], table.to_csv(index=False), "molsolveai_batch.csv", "text/csv", key="dl_batch_csv")
        st.download_button(t["dl_full"], json.dumps(res, ensure_ascii=False, indent=2), "molsolveai_batch_full.json",
                           "application/json", key="dl_batch_json")
        st.download_button(t["dl_html"], ultra.html_report(res, lang, t, config.TOX21_TASKS), "molsolveai_batch.html",
                           "text/html", key="dl_batch_html")
        st.subheader(t["detail"])
        for i, r in enumerate(res):
            with st.expander(f"{i+1}. {r['smiles']}  |  QED {r['rules']['qed']:.2f}  |  {t['conf']} {r['confidence']}%"):
                render(r, f"b{i}")
    if B and B["bad"]:
        st.warning(t["skipped"] + ": " + ", ".join(B["bad"][:20]))
    elif B is not None and not B["res"]:
        st.error(t["bad"])

with tab_cmp:
    st.write(t["cmp_help"])
    pick = st.multiselect(t["cmp_pick"], list(ultra.LIB), default=["Aspirin", "Ibuprofen", "Caffeine"], max_selections=4)
    extra = st.text_area(t["cmp_extra"], height=70)
    if st.button(t["cmp_go"]):
        ents = [(n, ultra.LIB[n]) for n in pick] + [(x.strip()[:18], x.strip()) for x in extra.splitlines() if x.strip()]
        out = []
        for n, x in ents[:4]:
            r = analyze(x)
            if r:
                out.append((n, r))
        st.session_state.cmp = out if len(out) >= 2 else None
        if len(out) < 2:
            st.error(t["cmp_need"])
    C = st.session_state.get("cmp")
    if C:
        st.pyplot(ultra.radar(C))
        cols = st.columns(len(C))
        for col, (n, r) in zip(cols, C):
            ov, _ = ultra.verdict(r, lang)
            col.markdown(f"**{ultra.DOT[ov]} {n}**")
            col.image(Draw.MolToImage(Chem.MolFromSmiles(r["smiles"]), size=(220, 180)))
        st.subheader(L("Ranking with reasons", "الترتيب مع المبررات"))
        for i_, x in enumerate(v7.rank_candidates(C, model, lang), 1):
            st.markdown(f"**{i_}. {x['name']}**")
            st.write(L("Strengths: ", "نقاط القوة: ") + (L(", ", "، ").join(x["good"]) or "-"))
            st.write(L("Limits: ", "القيود: ") + ("; ".join(x["bad"]) or "-"))
        st.caption(v7.RULE[lang])
        tbl = pd.DataFrame([{"name": n, **{k: v for k, v in flat(r).items() if k != "smiles"}} for n, r in C]).set_index("name").T.astype(str)
        st.dataframe(tbl, height=700)

with tab_opt:
    st.write(L("Pick a molecule and a goal. MolSolveAI generates virtual variants (add F, swap groups, ring N...) and ranks them by predicted improvement.",
               "اختر جزيئاً وهدفاً. يولّد MolSolveAI متغيرات افتراضية (إضافة F، استبدال مجموعات، نيتروجين في الحلقة...) ويرتبها حسب التحسن المتوقع."))
    o_name = st.selectbox(L("Starting molecule", "الجزيء الأساسي"), list(ultra.LIB), index=0)
    o_extra = st.text_input(L("...or your own SMILES (optional)", "...أو SMILES خاص بك (اختياري)"), "")
    goals = {"sol": L("Higher solubility", "ذوبانية أعلى"), "tox": L("Lower toxicity signals", "إشارات سمّية أقل"),
             "bbb_up": L("Better brain penetration", "عبور أفضل للدماغ"), "bbb_down": L("Less brain penetration", "عبور أقل للدماغ"),
             "qed": L("Higher drug-likeness (QED)", "شبه دواء أعلى (QED)"),
             "balanced": L("Balanced (QED + solubility + fewer tox flags + easier to make)", "متوازن (QED وذوبانية وسمّية أقل وتصنيع أسهل)")}
    goal = st.selectbox(L("Goal", "الهدف"), list(goals), format_func=lambda k: goals[k])
    if st.button(L("Optimize", "حسّن")):
        base_smi = o_extra.strip() or ultra.LIB[o_name]
        if canonical(base_smi) is None:
            st.error(t["bad"])
        else:
            base, cands = ultra2.optimize(canonical(base_smi), analyze, goal)
            st.session_state.opt = (base, cands, goal)
    O = st.session_state.get("opt")
    if O:
        base, cands, g_ = O
        st.markdown(f"**{L('Starting point', 'نقطة البداية')}:** `{base['smiles']}`")
        if not cands or cands[0][0] <= 0:
            st.info(L("No variant improved this goal.", "لا يوجد متغير يحسّن هذا الهدف."))
        for rank, (delta, name_, s_, r_) in enumerate(cands, 1):
            if delta <= 0:
                continue
            c0, c1 = st.columns([1, 2])
            c0.image(Draw.MolToImage(Chem.MolFromSmiles(s_), size=(240, 190)))
            ov_, _ = ultra.verdict(r_, lang)
            c1.markdown(f"**#{rank} {ultra.DOT[ov_]} {name_}**  \n`{s_}`")
            bt = base["tasks"]
            def dl(a, b, f="{:+.2f}"):
                return f.format(a - b)
            c1.write(f"LogS {r_['tasks']['LogS']['value']:.2f} ({dl(r_['tasks']['LogS']['value'], bt['LogS']['value'])}) | BBB {r_['tasks']['BBB']['prob']*100:.0f}% ({dl(r_['tasks']['BBB']['prob']*100, bt['BBB']['prob']*100, '{:+.0f}')}) | QED {r_['rules']['qed']:.2f} ({dl(r_['rules']['qed'], base['rules']['qed'])}) | tox flags {len(r_['tox_flags'])} ({len(r_['tox_flags']) - len(base['tox_flags']):+d})")
            c1.write(f"SA {r_['sa']} ({v6.sa_text(r_['sa'], lang == 'ar')}) | {v6.AD_TXT[r_['ad']['band']][0]} {L('domain', 'النطاق')}")
            if rank <= 3:
                c1.caption(L("Why this step: ", "لماذا هذه الخطوة: ") + name_ + " → " + L("compare before/after above; the change that moves your goal most with the smallest edit.", "قارن قبل/بعد أعلاه؛ أكثر تغيير يحرّك هدفك بأصغر تعديل."))
            c1.write(f"{L('Improvement score', 'درجة التحسن')}: +{delta:.2f}")
        st.warning(L("These are virtual ideas scored by an estimate model. They are NOT tested, may be hard or impossible to make, and may be worse in reality.",
                     "هذه أفكار افتراضية يقيّمها نموذج تقديري. لم تُختبر، وقد يصعب أو يستحيل تصنيعها، وقد تكون أسوأ في الواقع."))

with tab_teach:
    st.write(L("Make a printable worksheet and answer key from any molecules. Answers are exact RDKit calculations.",
               "أنشئ ورقة عمل ومفتاح إجابة قابلين للطباعة من أي جزيئات. الإجابات حسابات RDKit دقيقة."))
    tp = st.multiselect(L("Pick molecules (up to 5)", "اختر جزيئات (حتى 5)"), list(ultra.LIB), default=["Aspirin", "Caffeine"], max_selections=5, key="tp")
    tx = st.text_area(L("Extra SMILES, one per line (optional)", "SMILES إضافية، سطر لكل جزيء (اختياري)"), height=60, key="tx")
    sch = st.text_input(L("School / class name (optional)", "اسم المدرسة أو الصف (اختياري)"), "")
    if st.button(L("Create worksheet", "أنشئ الورقة")):
        its = []
        for n_ in tp:
            its.append((n_, ultra.LIB[n_], druglike.rules(ultra.LIB[n_])))
        for x in [y.strip() for y in tx.splitlines() if y.strip()][:5]:
            c_ = canonical(x)
            if c_:
                its.append((c_[:20], c_, druglike.rules(c_)))
        st.session_state.sheet = v6.teacher_sheet(its, lang, sch) if its else None
    if st.session_state.get("sheet"):
        st.success(L("Ready.", "جاهزة."))
        st.download_button(L("⬇️ Download worksheet + answer key (HTML, print as PDF)", "⬇️ تنزيل الورقة ومفتاح الإجابة (HTML، اطبعها PDF)"), st.session_state.sheet,
                           "molsolveai_worksheet.html", "text/html", key="dl_sheet")

with tab_quiz:
    st.write(L("Test your chemistry intuition. Answers use exact RDKit calculations, not AI guesses.",
               "اختبر حدسك الكيميائي. الإجابات من حسابات RDKit الدقيقة وليست تخميناً من الذكاء الاصطناعي."))
    st.session_state.setdefault("score", [0, 0])
    if "q" not in st.session_state:
        st.session_state.q = ultra2.quiz_question(ultra.LIB)
    nm, sm, kind, thr, truth, fact = st.session_state.q
    texts = {"mw": L(f"Is the molecular weight above {thr}?", f"هل الوزن الجزيئي أكبر من {thr}؟"),
             "logp": L(f"Is LogP above {thr}?", f"هل LogP أكبر من {thr}؟"),
             "hbd": L(f"Does it have more than {thr} hydrogen-bond donors?", f"هل فيه أكثر من {thr} مانح هيدروجين؟"),
             "lip": L("Does it follow Lipinski's rule of five (0 violations)?", "هل يطابق قاعدة ليبنسكي الخمسية (0 مخالفات)؟")}
    st.image(Draw.MolToImage(Chem.MolFromSmiles(sm), size=(320, 240)))
    st.subheader(texts[kind])
    cA, cB, cC = st.columns(3)
    ans = None
    if cA.button(L("Yes", "نعم"), key="qy"): ans = True
    if cB.button(L("No", "لا"), key="qn"): ans = False
    if cC.button(L("Next molecule", "جزيء آخر"), key="qx"):
        st.session_state.q = ultra2.quiz_question(ultra.LIB); st.session_state.pop("qres", None); st.rerun()
    if ans is not None:
        ok = ans == truth
        st.session_state.score[0] += int(ok); st.session_state.score[1] += 1
        st.session_state.qres = (ok, nm, fact)
    if st.session_state.get("qres"):
        ok, nm_, fact_ = st.session_state.qres
        (st.success if ok else st.error)(f"{'✅' if ok else '❌'} {nm_}: {fact_}")
    st.metric(L("Score", "النتيجة"), f"{st.session_state.score[0]} / {st.session_state.score[1]}")

with tab_card:
    st.markdown(v6.MODEL_CARD[lang])
    st.table(pd.DataFrame({"task": list(model.test_metrics), "score": [", ".join(f"{k} {v:.2f}" for k, v in m.items()) for m in model.test_metrics.values()]}))
    bl = v7.baseline()
    if bl:
        st.subheader(L("Comparison with simple baselines", "مقارنة مع خطوط أساس بسيطة"))
        rows = []
        for tk, v in bl["tasks"].items():
            rf = v.get("random_forest", {}); rg = v.get("ridge_descriptors")
            met = "R2" if "R2" in rf else "AUC"
            rows.append({L("task", "الاختبار"): tk, L("our reported", "رقمنا المعلن"): f"{model.test_metrics.get(tk, {}).get(met, float('nan')):.2f}",
                         L("random forest (baseline)", "غابة عشوائية (أساس)"): f"{rf[met]:.2f}",
                         L("ridge (baseline)", "ريدج (أساس)"): f"{rg['R2']:.2f}" if rg else "-", L("metric", "المقياس"): met})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.warning(v7.BASE_NOTE[lang])
    st.subheader(L("Transparent check on well-known drugs (brain penetration)", "فحص شفاف على أدوية معروفة (عبور الدماغ)"))
    st.caption(L("Expected values are textbook knowledge. Many of these drugs may be in the training data, so a match here is NOT proof of accuracy on new molecules. Check the similarity column.",
                 "القيم المتوقعة معرفة عامة في الكتب. كثير من هذه الأدوية قد يكون في بيانات التدريب، فالتطابق هنا ليس دليلاً على الدقة في جزيئات جديدة. انظر عمود التشابه."))
    if st.button(L("Run the check", "شغّل الفحص")):
        rows, hit = v6.check_known(analyze, ultra.LIB)
        st.metric(L("Matches", "التطابق"), f"{hit} / {len(rows)}")
        st.dataframe(pd.DataFrame(rows), hide_index=True)

with tab3:
    st.subheader(t["plans_head"])
    st.info(t["plans_note"])
    st.table(pd.DataFrame([{"Plan": p["name"], t["price"]: p["price"], "Single/session": p["single"], "Batch rows": p["batch"],
                            "API": "-" if k == "free" else "planned"} for k, p in PLANS.items()]))
    st.subheader(t["api_head"]); st.write(t["api_note"])

with tab4:
    st.markdown(f"### {L('About the brand', 'عن العلامة')}")
    st.write(L("MolSolveAI is a product of Manarat Al-Ibdaa: making molecular science visible, explainable and fun with AI, in Arabic and English.",
               "MolSolveAI من علامة منارة الإبداع: نجعل علم الجزيئات مرئياً ومفهوماً وممتعاً بالذكاء الاصطناعي، بالعربية والإنجليزية."))
    st.write(L("What makes it different: it explains WHY (atom-level maps), suggests better molecules for a goal (Optimizer), compares molecules visually, and teaches with a quiz.",
               "ما يميّزه: يشرح لماذا (خرائط على مستوى الذرات)، ويقترح جزيئات أفضل لهدف محدد (المحسّن)، ويقارن الجزيئات بصرياً، ويعلّم بتحدٍ تفاعلي."))
    st.write(t["about"]); st.success(t["acc"]); st.caption(t["disc"])


with tab_ver:
    st.write(L("Upload a dossier JSON made by this app. We recompute it from its molecule and compare the report ID and model fingerprint.", "ارفع ملف JSON لملف أدلة صدر من هذا التطبيق. نعيد حسابه من جزيئه ونقارن رقم التقرير وبصمة النموذج."))
    vf = st.file_uploader(L("Dossier JSON", "ملف الأدلة JSON"), type=["json"], key="ver_up")
    if vf is not None:
        try:
            old = json.load(vf)
            old = old[0] if isinstance(old, list) else old
            r2 = analyze(old["identity"]["canonical_smiles"])
            new = dossier.build(r2, model, None, lang)
            same_model = old["reproducibility"]["model_sha256"] == new["reproducibility"]["model_sha256"]
            same_id = old["report_id"] == new["report_id"]
            (st.success if same_id else st.warning)(L("Report ID matches: results reproduced exactly.", "رقم التقرير مطابق: النتائج أُعيدت كما هي.") if same_id else L("Report ID differs.", "رقم التقرير مختلف."))
            if not same_id:
                st.write(L("Same model fingerprint: ", "بصمة النموذج نفسها: ") + (L("yes (the file may have been edited or made by another app version)", "نعم (ربما عُدّل الملف أو صدر من نسخة تطبيق أخرى)") if same_model else L("no (made with a different model version)", "لا (صدر بنسخة نموذج مختلفة)")))
            st.caption(f"{old['report_id']} / {new['report_id']}")
        except Exception:
            st.error(L("This file is not a valid dossier JSON.", "هذا الملف ليس ملف أدلة صالحاً."))
