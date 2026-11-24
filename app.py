"""الواجهة: streamlit run app.py"""
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import config
from chem import canonical

st.set_page_config(page_title="MolSolveAI", page_icon="🧪", layout="centered")
st.title("🧪 MolSolveAI")
st.warning(config.DISCLAIMER)

import infer
if not infer.NPZ_PATH.exists():
    st.error("لم يتم تجهيز النموذج بعد. شغّل:  python train.py ثم python export_weights.py")
    st.stop()


@st.cache_resource
def get():
    return infer.load()


model = get()
EXAMPLES = {"أسبرين": "CC(=O)Oc1ccccc1C(=O)O", "كافيين": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
            "إيبوبروفين": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "بنزين": "c1ccccc1"}
choice = st.selectbox("أمثلة جاهزة", ["(اكتب بنفسك)"] + list(EXAMPLES))
smi = st.text_input("اكتب صيغة SMILES للجزيء", EXAMPLES.get(choice, "CC(=O)Oc1ccccc1C(=O)O"))

if st.button("حلّل"):
    c = canonical(smi)
    if c is None:
        st.error("صيغة SMILES غير صحيحة. جرّب أحد الأمثلة.")
        st.stop()
    r = infer.predict(c, model)
    st.image(Draw.MolToImage(Chem.MolFromSmiles(c), size=(380, 300)))
    st.subheader("خصائص محسوبة بدقة (RDKit)")
    st.table({k: [v] for k, v in r["physchem"].items()})
    st.caption(f"مخالفات قاعدة ليبنسكي: {r['lipinski_violations']} (0 = ممتاز كدواء فموي)")
    st.subheader("تنبؤات الذكاء الاصطناعي")
    t = r["tasks"]
    st.write(f"**الذوبانية LogS:** {t['LogS']['value']:.2f} ± {t['LogS']['uncertainty']:.2f}")
    bbb = t["BBB"]
    st.write(f"**عبور الحاجز الدموي الدماغي:** {'نعم' if bbb['prob']>0.5 else 'لا'} "
             f"(احتمال {bbb['prob']*100:.0f}% ± {bbb['uncertainty']*100:.0f})")
    st.markdown("**اختبارات السمّية (Tox21): احتمال النشاط**")
    rows = {k: [f"{v['prob']*100:.0f}% ± {v['uncertainty']*100:.0f}"] for k, v in t.items() if k in config.TOX21_TASKS}
    st.table(rows)
    st.caption("± = عدم اليقين (تفاوت التنبؤ عبر 30 تشغيلاً). كلما كبر قلّت الثقة. "
               "")
    st.warning(config.DISCLAIMER)
