# streamlit_app_final_ml_uz_full.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shap

# Page config
st.set_page_config(
    page_title="💼 Biznes Bankrotlik va Kredit Tavsiyasi (ML)",
    layout="wide",
    page_icon="💰"
)

st.title("💼 Biznes Bankrotlik va Kredit Tavsiyasi (O‘zbekiston) - ML Versiya")
st.write("""
Ushbu ilova orqali siz bir yoki bir nechta biznesning **bankrotlik xavfini** baholashingiz va
xavf past bo‘lsa **tavsiya qilinadigan kredit miqdorini** ML yordamida ko‘rishingiz mumkin.
""")
st.markdown("---")

# ---------------------------
# Bankrotlik scoring funksiyasi
# ---------------------------
def hisobla_bankrotlik(row):
    risk = 0
    if row["sof_foyda"] < 0: risk += 3
    if row["foyda_margini"] < 0.05: risk +=1
    if row["kech_tolovlar_soni"] > 3: risk +=2
    if row["soliq_qarzi_uzs"] > 15_000_000: risk +=2
    if row["kredit_yuklama"] > 0.4: risk +=2
    if row["biznes_yoshi_yil"] < 2: risk +=1
    if row.get("sohasi","") == "Savdo": risk +=1
    if row.get("hudud","") == "Viloyat": risk +=1
    return 1 if risk >=5 else 0, risk

# ---------------------------
# Foydalanuvchi input
# ---------------------------
mode = st.radio("Qanday kiritmoqchisiz?", ["Bitta biznes", "CSV fayl orqali ko‘p biznes"])

# ---------------------------
# 1️⃣ Individual biznes
# ---------------------------
if mode == "Bitta biznes":
    with st.form("individual_form"):
        oylik_daromad = st.number_input("Oylik daromad (UZS)", min_value=0, value=50_000_000, step=1_000_000)
        oylik_xarajat = st.number_input("Oylik xarajat (UZS)", min_value=0, value=30_000_000, step=1_000_000)
        kredit_miqdori = st.number_input("Hozirgi kredit miqdori (UZS)", min_value=0, value=0, step=1_000_000)
        kredit_oylik_tolovi = st.number_input("Oylik kredit to‘lovi (UZS)", min_value=0, value=0, step=500_000)
        soliq_qarzi = st.number_input("Soliq qarzi (UZS)", min_value=0, value=0, step=500_000)
        biznes_yoshi = st.number_input("Biznes yoshi (yil)", min_value=0, value=3)
        kech_tolovlar = st.number_input("Kechiktirilgan to‘lovlar soni", min_value=0, value=1)
        sohasi = st.selectbox("Sohasi", ["Savdo", "Xizmat", "Ishlab chiqarish"])
        hudud = st.selectbox("Hudud", ["Toshkent", "Viloyat"])
        kredit_vaqt = st.radio("Kredit tavsiyasi:", ["Oyiga", "Yiliga"])
        submitted = st.form_submit_button("💻 Hisoblash")

    if submitted:
        sof_foyda = oylik_daromad - oylik_xarajat
        foyda_margini = sof_foyda / oylik_daromad if oylik_daromad>0 else 0
        qarz_daromadga_nisbati = kredit_miqdori / oylik_daromad if oylik_daromad>0 else 0
        kredit_yuklama = kredit_oylik_tolovi / oylik_daromad if oylik_daromad>0 else 0

        row = {
            "sof_foyda": sof_foyda,
            "foyda_margini": foyda_margini,
            "kredit_yuklama": kredit_yuklama,
            "biznes_yoshi_yil": biznes_yoshi,
            "qarz_daromadga_nisbati": qarz_daromadga_nisbati,
            "kech_tolovlar_soni": kech_tolovlar,
            "soliq_qarzi_uzs": soliq_qarzi,
            "sohasi": sohasi,
            "hudud": hudud
        }
        bankrotlik_riski, risk_ball = hisobla_bankrotlik(row)
        risk_prosent = min(risk_ball*10,100)

        # Tavsiya kredit minimal/maximal
        if bankrotlik_riski==0:
            tavsiya_kredit_min = max(int(sof_foyda*0.6*0.8 - kredit_oylik_tolovi),0)
            tavsiya_kredit_max = max(int(sof_foyda*0.6*1.2 - kredit_oylik_tolovi),0)
            if kredit_vaqt=="Yiliga":
                tavsiya_kredit_min *= 12
                tavsiya_kredit_max *= 12
        else:
            tavsiya_kredit_min = tavsiya_kredit_max = 0

        # Natija chiqarish
        st.header("📊 Natija")
        if bankrotlik_riski==0:
            st.success(f"✅ Biznes xavfsiz! Bankrotlik xavfi: {risk_prosent:.0f}%")
            st.info(f"Tavsiya qilinadigan kredit: {tavsiya_kredit_min:,} – {tavsiya_kredit_max:,} UZS ({kredit_vaqt.lower()})")
        else:
            st.error(f"⚠️ Biznes xavfli! Bankrotlik xavfi: {risk_prosent:.0f}%")
            st.warning("Yangi kredit tavsiya qilinmaydi")
            st.info("Sababi: sof foyda past, kechiktirilgan to‘lovlar yoki soliq qarzi yuqori bo‘lishi mumkin")

        # ---------------------------
        # Faktorlar rolini vizualizatsiya
        # ---------------------------
        feature_importance = pd.DataFrame({
            "Faktor": ["Sof foyda","Foyda margini","Kredit yuklama","Biznes yoshi","Qarz/daromad","Kech to‘lovlar","Soliq qarzi","Sohasi","Hudud"],
            "Rol (%)": [3,1,2,1,1,2,2,1,1]  # risk_ball asosida taxminiy %
        }).sort_values("Rol (%)", ascending=False)

        st.subheader("📈 Faktorlar bankrotlikda qancha rol o'ynaydi")
        fig, ax = plt.subplots()
        sns.barplot(x="Rol (%)", y="Faktor", data=feature_importance, palette="viridis", ax=ax)
        st.pyplot(fig)

# ---------------------------
# 2️⃣ CSV batch biznes
# ---------------------------
else:
    uploaded_file = st.file_uploader("CSV faylni yuklang", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df["sof_foyda"] = df["oylik_daromad_uzs"] - df["oylik_xarajat_uzs"]
        df["foyda_margini"] = df["sof_foyda"]/df["oylik_daromad_uzs"].replace(0,np.nan)
        df["qarz_daromadga_nisbati"] = df["kredit_miqdori_uzs"]/df["oylik_daromad_uzs"].replace(0,np.nan)
        df["kredit_yuklama"] = df["kredit_oylik_tolovi"]/df["oylik_daromad_uzs"].replace(0,np.nan)

        df["bankrotlik_riski"], df["risk_ball"] = zip(*df.apply(hisobla_bankrotlik, axis=1))
        df["risk_prosent"] = df["risk_ball"].clip(0,100)

        # Tavsiya kredit
        df["tavsiya_kredit_min"] = np.where(df["bankrotlik_riski"]==0, (df["sof_foyda"]*0.6*0.8 - df["kredit_oylik_tolovi"]).clip(0),0)
        df["tavsiya_kredit_max"] = np.where(df["bankrotlik_riski"]==0, (df["sof_foyda"]*0.6*1.2 - df["kredit_oylik_tolovi"]).clip(0),0)

        # ML modeli
        features = ["sof_foyda","foyda_margini","kredit_yuklama","biznes_yoshi_yil",
                    "qarz_daromadga_nisbati","kech_tolovlar_soni","soliq_qarzi_uzs"]
        X = df[features].fillna(0)
        y = df["bankrotlik_riski"]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        y_pred = rf_model.predict(X_test_scaled)
        rf_acc = accuracy_score(y_test, y_pred)

        st.header("📊 ML Model Performance")
        st.metric("Random Forest ACC", f"{rf_acc:.2f}")

        # SHAP interaktiv
        st.subheader("🔹 Kredit rad etish sabablari (SHAP interaktiv)")
        st.write("Pastdagi slider orqali biznesni tanlab, faktorlar roli ko‘rish mumkin:")

        if len(X_test_scaled) > 0:
            idx_slider = st.slider("Biznes indeksini tanlang", 0, len(X_test_scaled)-1, 0)
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_test_scaled)
            shap.initjs()
            shap_fig = shap.force_plot(
                explainer.expected_value[1],
                shap_values[1][idx_slider,:],
                pd.DataFrame(X_test_scaled, columns=features).iloc[idx_slider,:],
                matplotlib=True
            )
            st.pyplot(shap_fig)

        # Natijalar jadval
        st.header("📊 Natijalar jadval")
        st.dataframe(df[["sohasi","hudud","sof_foyda","bankrotlik_riski","risk_prosent","tavsiya_kredit_min","tavsiya_kredit_max"]])