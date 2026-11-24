"""MolSolveAI web app. Run: streamlit run app.py"""
import io, json, os
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import config, infer, druglike, ultra, ultra2
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
    return r


tab1, tab2, tab_cmp, tab_opt, tab_quiz, tab3, tab4 = st.tabs([t["tab_single"], t["tab_batch"], t["tab_cmp"], L("Optimizer", "محسّن الجزيء"), L("Challenge", "تحدي الكيمياء"), t["tab_plans"], t["tab_about"]])

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
        res, bad = [], []
        prog = st.progress(0.0)
        for i, x in enumerate(smiles):
            r = analyze(x)
            if r is None:
                bad.append(x)
            else:
                res.append(r)
            prog.progress((i + 1) / max(1, len(smiles)))
        st.session_state.batch = {"res": res, "bad": bad}
    B = st.session_state.get("batch")
    if B and B["res"]:
        res = B["res"]
        st.subheader(t["summary"])
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(t["rows"], len(res)); k2.metric(t["n_dl"], sum(r["rules"]["lipinski_viol"] == 0 for r in res))
        k3.metric(t["n_alert"], sum(bool(r["alerts"]) for r in res)); k4.metric(t["n_tox"], sum(bool(r["tox_flags"]) for r in res))
        best = max(res, key=lambda r: r["rules"]["qed"])
        st.write(f"{t['best']}: `{best['smiles']}` (QED {best['rules']['qed']:.2f})")
        sort_by = st.selectbox(t["sort"], ["QED", "confidence", "LogS", "BBB_prob", "tox_flag_count"])
        table = pd.DataFrame([flat(r) for r in res]).sort_values(sort_by, ascending=(sort_by == "tox_flag_count")).reset_index(drop=True)
        st.dataframe(table)
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
        tbl = pd.DataFrame([{"name": n, **{k: v for k, v in flat(r).items() if k != "smiles"}} for n, r in C]).set_index("name").T.astype(str)
        st.dataframe(tbl, height=700)

with tab_opt:
    st.write(L("Pick a molecule and a goal. MolSolveAI generates virtual variants (add F, swap groups, ring N...) and ranks them by predicted improvement.",
               "اختر جزيئاً وهدفاً. يولّد MolSolveAI متغيرات افتراضية (إضافة F، استبدال مجموعات، نيتروجين في الحلقة...) ويرتبها حسب التحسن المتوقع."))
    o_name = st.selectbox(L("Starting molecule", "الجزيء الأساسي"), list(ultra.LIB), index=0)
    o_extra = st.text_input(L("...or your own SMILES (optional)", "...أو SMILES خاص بك (اختياري)"), "")
    goals = {"sol": L("Higher solubility", "ذوبانية أعلى"), "tox": L("Lower toxicity signals", "إشارات سمّية أقل"),
             "bbb_up": L("Better brain penetration", "عبور أفضل للدماغ"), "bbb_down": L("Less brain penetration", "عبور أقل للدماغ"),
             "qed": L("Higher drug-likeness (QED)", "شبه دواء أعلى (QED)")}
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
            c1.write(f"LogS {r_['tasks']['LogS']['value']:.2f} | BBB {r_['tasks']['BBB']['prob']*100:.0f}% | QED {r_['rules']['qed']:.2f} | tox flags {len(r_['tox_flags'])}")
            c1.write(f"{L('Improvement score', 'درجة التحسن')}: +{delta:.2f}")
        st.warning(L("These are virtual ideas scored by an estimate model. They are NOT tested, may be hard or impossible to make, and may be worse in reality.",
                     "هذه أفكار افتراضية يقيّمها نموذج تقديري. لم تُختبر، وقد يصعب أو يستحيل تصنيعها، وقد تكون أسوأ في الواقع."))

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
