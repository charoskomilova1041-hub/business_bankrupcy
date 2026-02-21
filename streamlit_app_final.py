# streamlit_app_final.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Page konfiguratsiyasi
# ---------------------------
st.set_page_config(
    page_title="💼 Biznes Bankrotlik va Kredit Tavsiyasi",
    layout="wide",
    page_icon="💰"
)

# ---------------------------
# 1️⃣ Sarlavha va ta’rif
# ---------------------------
st.title("💼 Biznes Bankrotlik va Kredit Tavsiyasi (O‘zbekiston)")
st.write("""
Ushbu ilova orqali siz yoki bir nechta biznesning **bankrotlik xavfini** baholashingiz va
xavf past bo‘lsa **tavsiya qilinadigan kredit miqdorini** ko‘rishingiz mumkin.
""")

st.markdown("---")

# ---------------------------
# 2️⃣ Foydalanuvchi variantlari: Individual yoki CSV
# ---------------------------
st.header("Biznes ma’lumotlarini kiritish:")

mode = st.radio("Qanday kiritmoqchisiz?", ["Bitta biznes", "CSV fayl orqali ko‘p biznes"])

if mode == "Bitta biznes":
    oylik_daromad = st.number_input("Oylik daromad (UZS)", min_value=0, value=50_000_000, step=1_000_000, help="Biznes oyiga qancha pul topadi")
    oylik_xarajat = st.number_input("Oylik xarajat (UZS)", min_value=0, value=30_000_000, step=1_000_000, help="Ish haqi, ijara, soliq va boshqa xarajatlar")
    kredit_miqdori = st.number_input("Hozirgi kredit miqdori (UZS)", min_value=0, value=0, step=1_000_000)
    kredit_oylik_tolovi = st.number_input("Oylik kredit to‘lovi (UZS)", min_value=0, value=0, step=500_000)
    soliq_qarzi = st.number_input("Soliq qarzi (UZS)", min_value=0, value=0, step=500_000)
    biznes_yoshi = st.number_input("Biznes yoshi (yil)", min_value=0, value=3)
    kech_tolovlar = st.number_input("Kechiktirilgan to‘lovlar soni", min_value=0, value=1)
    sohasi = st.selectbox("Sohasi", ["Savdo", "Xizmat", "Ishlab chiqarish"])
    hudud = st.selectbox("Hudud", ["Toshkent", "Viloyat"])

    # ---------------------------
    # Feature engineering
    # ---------------------------
    sof_foyda = oylik_daromad - oylik_xarajat
    foyda_margini = sof_foyda / oylik_daromad if oylik_daromad > 0 else 0
    qarz_daromadga_nisbati = kredit_miqdori / oylik_daromad if oylik_daromad > 0 else 0
    kredit_yuklama = kredit_oylik_tolovi / oylik_daromad if oylik_daromad > 0 else 0

    # Bankrotlik scoring
    risk_ball = 0
    if sof_foyda < 0: risk_ball += 3
    if foyda_margini < 0.05: risk_ball +=1
    if kech_tolovlar > 3: risk_ball +=2
    if soliq_qarzi > 15_000_000: risk_ball +=2
    if kredit_yuklama > 0.4: risk_ball +=2
    if biznes_yoshi < 2: risk_ball +=1
    if qarz_daromadga_nisbati >1: risk_ball +=1

    bankrotlik_riski = 1 if risk_ball >= 5 else 0
    risk_prosent = min(risk_ball*10,100)

    # Tavsiya kredit
    if bankrotlik_riski == 0:
        tavsiya_kredit = max(int(sof_foyda*0.6 - kredit_oylik_tolovi), 0)
    else:
        tavsiya_kredit = 0

    # ---------------------------
    # Natijalarni chiqarish
    # ---------------------------
    st.header("📊 Natija")
    if bankrotlik_riski == 0:
        st.success(f"✅ Biznes xavfsiz! Bankrotlik xavfi: {risk_prosent:.0f}%")
        st.info(f"Tavsiya qilinadigan kredit: {tavsiya_kredit:,} UZS")
    else:
        st.error(f"⚠️ Biznes xavfli! Bankrotlik xavfi: {risk_prosent:.0f}%")
        st.warning("Yangi kredit tavsiya qilinmaydi")

else:  # CSV batch
    uploaded_file = st.file_uploader("CSV faylni yuklang", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Feature engineering
        df["sof_foyda"] = df["oylik_daromad_uzs"] - df["oylik_xarajat_uzs"]
        df["foyda_margini"] = df["sof_foyda"] / df["oylik_daromad_uzs"].replace(0, np.nan)
        df["qarz_daromadga_nisbati"] = df["kredit_miqdori_uzs"] / df["oylik_daromad_uzs"].replace(0, np.nan)
        df["kredit_yuklama"] = df["kredit_oylik_tolovi"] / df["oylik_daromad_uzs"].replace(0, np.nan)

        # Bankrotlik scoring
        def hisobla_bankrotlik(row):
            risk = 0
            if row["sof_foyda"] < 0: risk += 3
            if row["foyda_margini"] < 0.05: risk +=1
            if row["kech_tolovlar_soni"] > 3: risk +=2
            if row["soliq_qarzi_uzs"] > 15_000_000: risk +=2
            if row["kredit_yuklama"] > 0.4: risk +=2
            if row["biznes_yoshi_yil"] < 2: risk +=1
            if row["qarz_daromadga_nisbati"] >1: risk +=1
            return 1 if risk >=5 else 0

        df["bankrotlik_riski"] = df.apply(hisobla_bankrotlik, axis=1)
        df["risk_prosent"] = df["bankrotlik_riski"]*100

        # Tavsiya kredit
        df["tavsiya_kredit"] = np.where(
            df["bankrotlik_riski"]==0,
            (df["sof_foyda"]*0.6 - df["kredit_oylik_tolovi"]).clip(lower=0),
            0
        )

        # ---------------------------
        # Natijalar jadval
        # ---------------------------
        st.header("📊 Natijalar jadval")
        st.dataframe(df[[
            "sohasi","hudud","oylik_daromad_uzs","oylik_xarajat_uzs",
            "sof_foyda","bankrotlik_riski","risk_prosent","tavsiya_kredit"
        ]])

        # ---------------------------
        # Vizualizatsiya
        # ---------------------------
        st.header("📈 Vizualizatsiya")
        # Bankrotlik tarqalishi
        st.subheader("Bankrotlik tarqalishi")
        fig1, ax1 = plt.subplots()
        sns.countplot(x="bankrotlik_riski", data=df, ax=ax1, palette=["green","red"])
        ax1.set_xticklabels(["Xavfsiz","Xavfli"])
        st.pyplot(fig1)

        # Sektorlar bo‘yicha xavf
        st.subheader("Sektorlar bo‘yicha xavf")
        fig2, ax2 = plt.subplots()
        df.groupby("sohasi")["bankrotlik_riski"].mean().plot(kind="bar", ax=ax2, color="orange")
        ax2.set_ylabel("O‘rtacha bankrotlik riski")
        st.pyplot(fig2)

        # Tavsiya kredit miqdori
        st.subheader("Tavsiya qilinadigan kredit (UZS)")
        fig3, ax3 = plt.subplots()
        df[df["tavsiya_kredit"]>0].plot.scatter(
            x="sof_foyda", y="tavsiya_kredit", c="bankrotlik_riski",
            colormap="coolwarm", ax=ax3
        )
        ax3.set_xlabel("Sof foyda (UZS)")
        ax3.set_ylabel("Tavsiya kredit (UZS)")
        st.pyplot(fig3)
