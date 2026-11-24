"""MolSolveAI web app. Run: streamlit run app.py"""
import io, json, os
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import config, infer, druglike
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

st.title("🧪 " + t["title"])
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


tab1, tab2, tab3, tab4 = st.tabs([t["tab_single"], t["tab_batch"], t["tab_plans"], t["tab_about"]])

def render(r, key):
    R, ts = r["rules"], r["tasks"]
    c0, c1 = st.columns([1, 2])
    c0.image(Draw.MolToImage(Chem.MolFromSmiles(r["smiles"]), size=(300, 240)))
    m1, m2 = c1.columns(2)
    m1.metric(t["conf"], f"{r['confidence']}%", help=t["conf_help"])
    m2.metric("QED", f"{R['qed']:.2f}", help=t["qed"])
    c1.code(r["smiles"])
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
                               "± %": [round(ts[k]["uncertainty"] * 100) for k in config.TOX21_TASKS]}), hide_index=True)
    st.bar_chart(pd.Series({k: ts[k]["prob"] for k in config.TOX21_TASKS}))
    st.write(("**" + t["risk"] + ":** " + ", ".join(r["tox_flags"])) if r["tox_flags"] else t["risk0"])
    st.download_button(t["dl_json"], json.dumps(r, ensure_ascii=False, indent=2), f"molsolveai_{key}.json",
                       "application/json", key=f"dl_{key}")


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
    choice = st.selectbox(t["examples"], [t["own"]] + list(EXAMPLES))
    smi = st.text_input(t["smiles"], EXAMPLES.get(choice, EXAMPLES["Aspirin"]))
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
        st.subheader(t["detail"])
        for i, r in enumerate(res):
            with st.expander(f"{i+1}. {r['smiles']}  |  QED {r['rules']['qed']:.2f}  |  {t['conf']} {r['confidence']}%"):
                render(r, f"b{i}")
    if B and B["bad"]:
        st.warning(t["skipped"] + ": " + ", ".join(B["bad"][:20]))
    elif B is not None and not B["res"]:
        st.error(t["bad"])

with tab3:
    st.subheader(t["plans_head"])
    st.info(t["plans_note"])
    st.table(pd.DataFrame([{"Plan": p["name"], t["price"]: p["price"], "Single/session": p["single"], "Batch rows": p["batch"],
                            "API": "-" if k == "free" else "planned"} for k, p in PLANS.items()]))
    st.subheader(t["api_head"]); st.write(t["api_note"])

with tab4:
    st.write(t["about"]); st.success(t["acc"]); st.caption(t["disc"])
