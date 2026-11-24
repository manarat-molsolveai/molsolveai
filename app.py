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
                R = r["rules"]
                st.image(Draw.MolToImage(Chem.MolFromSmiles(r["smiles"]), size=(380, 300)))
                c1, c2, c3 = st.columns(3)
                c1.metric(t["conf"], f"{r['confidence']}%", help=t["conf_help"])
                c2.metric("QED", f"{R['qed']:.2f}", help=t["qed"])
                c3.metric("Lipinski", f"{R['lipinski_viol']} {t['viol']}")
                st.subheader(t["props"])
                st.table(pd.DataFrame({"MW": [round(R["mw"], 2)], "LogP": [round(R["logp"], 2)], "TPSA": [round(R["tpsa"], 1)],
                                       "HBD": [R["hbd"]], "HBA": [R["hba"]], "RotB": [R["rot"]], "Rings": [R["rings"]]}))
                st.subheader(t["rules"])
                st.write(f"Lipinski: {R['lipinski_viol']} | Veber: {R['veber_viol']} | Ghose: {R['ghose_viol']} {t['viol']}")
                st.write(f"**{t['alerts']}:** " + (", ".join(r["alerts"]) if r["alerts"] else t["none"]))
                st.subheader(t["ai"])
                ts = r["tasks"]
                st.write(f"**{t['logs']}:** {ts['LogS']['value']:.2f} ± {ts['LogS']['uncertainty']:.2f}")
                b = ts["BBB"]
                st.write(f"**{t['bbb']}:** {t['yes'] if b['prob'] > .5 else t['no']} ({b['prob']*100:.0f}% ± {b['uncertainty']*100:.0f})")
                st.markdown(f"**{t['tox']}**")
                st.bar_chart(pd.Series({k: ts[k]["prob"] for k in config.TOX21_TASKS}))
                st.write(("**" + t["risk"] + ":** " + ", ".join(r["tox_flags"])) if r["tox_flags"] else t["risk0"])
                st.download_button(t["dl_json"], json.dumps(r, ensure_ascii=False, indent=2), "molsolveai_report.json", "application/json")
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
            smiles = [str(s) for s in df[col].dropna()]
        smiles += [s.strip() for s in pasted.splitlines() if s.strip()]
        cap = plan["batch"]
        if len(smiles) > cap:
            st.info(t["batch_cap"].format(n=cap)); smiles = smiles[:cap]
        rows, bad = [], 0
        prog = st.progress(0.0)
        for i, s in enumerate(smiles):
            r = analyze(s, passes=10)
            if r is None:
                bad += 1
            else:
                R, ts = r["rules"], r["tasks"]
                rows.append({"smiles": r["smiles"], "MW": round(R["mw"], 2), "LogP": round(R["logp"], 2), "TPSA": round(R["tpsa"], 1),
                             "QED": round(R["qed"], 3), "Lipinski_viol": R["lipinski_viol"], "alerts": len(r["alerts"]),
                             "LogS": round(ts["LogS"]["value"], 2), "BBB_prob": round(ts["BBB"]["prob"], 3),
                             "tox_flags": ";".join(r["tox_flags"]), "confidence": r["confidence"]})
            prog.progress((i + 1) / max(1, len(smiles)))
        if rows:
            out = pd.DataFrame(rows)
            st.write(f"{t['rows']}: {len(rows)} | {t['skipped']}: {bad}")
            st.dataframe(out)
            st.download_button(t["dl_csv"], out.to_csv(index=False), "molsolveai_batch.csv", "text/csv")
        elif smiles:
            st.error(t["bad"])

with tab3:
    st.subheader(t["plans_head"])
    st.info(t["plans_note"])
    st.table(pd.DataFrame([{"Plan": p["name"], t["price"]: p["price"], "Single/session": p["single"], "Batch rows": p["batch"],
                            "API": "-" if k == "free" else "planned"} for k, p in PLANS.items()]))
    st.subheader(t["api_head"]); st.write(t["api_note"])

with tab4:
    st.write(t["about"]); st.success(t["acc"]); st.caption(t["disc"])
