# app.py — FINAL (Minimal only; No Advanced; No What-if)
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="💼 Biznes Bankrotlik Risk Scoring (UZ)",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Biznes Bankrotlik Risk Scoring (O‘zbekiston)")
st.write("Minimal inputlar bilan **ML risk ehtimoli** va **bankcha kredit tavsiyasi** chiqariladi.")
st.caption("Yakuniy qaror: **Hard Rules (stress profit & DSCR)** + ML ehtimol (izoh sifatida).")
st.divider()

# =========================
# Load artifacts
# =========================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("rf_bankrot_model.pkl")
        features = joblib.load("model_features.pkl")
        threshold = joblib.load("best_threshold.pkl")
        return model, features, float(threshold)
    except Exception as e:
        st.error(
            "Model fayllarini yuklab bo‘lmadi. Quyidagilar app.py bilan bir papkada bo‘lsin:\n"
            "- rf_bankrot_model.pkl\n- model_features.pkl\n- best_threshold.pkl\n\n"
            f"Xatolik: {e}"
        )
        st.stop()

model, FEATURES, THRESHOLD = load_artifacts()

# =========================
# Helpers
# =========================
def estimate_monthly_payment(principal: float, annual_rate: float, months: int) -> float:
    """Annuitet to'lov (PMT). annual_rate=0.32 -> 32%/yil"""
    if principal <= 0 or months <= 0:
        return 0.0
    r = annual_rate / 12.0
    if r <= 0:
        return principal / months
    return principal * (r * (1 + r) ** months) / ((1 + r) ** months - 1)

def max_principal_from_payment(target_payment: float, annual_rate: float, months: int) -> float:
    """Annuitet formulani teskari ishlatib: PMT -> Principal."""
    if target_payment <= 0 or months <= 0:
        return 0.0
    r = annual_rate / 12.0
    if r <= 0:
        return target_payment * months
    return target_payment * ((1 + r) ** months - 1) / (r * (1 + r) ** months)

def build_row(d: dict) -> pd.DataFrame:
    """
    Minimal user input -> 1-row dataframe + feature engineering + align columns.
    Eslatma: trainingdagi FEATURES ro‘yxati bilan mos bo‘lishi shart.
    """
    df = pd.DataFrame([d])

    # Feature engineering (minimal)
    df["foyda"] = df["oylik_daromad"] - df["oylik_xarajat"]
    df["foyda_marjasi"] = df["foyda"] / df["oylik_daromad"].clip(lower=1)
    df["qarz_nisbati"] = df["qarz_miqdori"] / df["oylik_daromad"].clip(lower=1)
    df["xarajat_nisbati"] = df["oylik_xarajat"] / df["oylik_daromad"].clip(lower=1)
    df["kredit_yuklama"] = df["kredit_oylik_tolovi"] / df["oylik_daromad"].clip(lower=1)

    # Align columns (IMPORTANT)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURES]
    return df

def safe_predict_proba(model, X_row: pd.DataFrame) -> float:
    if not hasattr(model, "predict_proba"):
        st.error("Model predict_proba() ni qo‘llab-quvvatlamaydi. Probabilistik model kerak.")
        st.stop()
    p = float(model.predict_proba(X_row)[0, 1])
    return float(np.clip(p, 0.0, 1.0))

def risk_badge(prob: float) -> str:
    if prob < 0.25:
        return "🟢 PAST XAVF"
    elif prob < 0.55:
        return "🟡 O‘RTA XAVF"
    return "🔴 YUQORI XAVF"

def draw_gauge(prob: float):
    prob = float(np.clip(prob, 0.0, 1.0))
    fig, ax = plt.subplots(figsize=(7, 1.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.hlines(0.5, 0, 1, linewidth=18)
    ax.vlines(prob, 0.25, 0.75, linewidth=4)
    ax.text(prob, 0.88, f"{prob*100:.1f}%", ha="center", va="bottom")
    ax.text(0.0, 0.12, "0%", ha="left")
    ax.text(1.0, 0.12, "100%", ha="right")
    return fig

def explain_reasons_minimal(stress_profit: float, dscr: float, mavsumiylik: int, prob: float,
                           daromad: float, xarajat: float, qarz: float, monthly_payment: float):
    """
    Minimal, tushunarli sabablar (rules).
    """
    reasons = []
    marja = (daromad - xarajat) / max(daromad, 1)
    qarz_n = qarz / max(daromad, 1)
    yuk = monthly_payment / max(daromad, 1)

    if mavsumiylik == 1:
        reasons.append("Mavsumiylik bor: stress-testda daromad 25% kamaytiriladi.")

    if stress_profit <= 0:
        reasons.append("Stress-holatda sof foyda manfiy/0: xarajat stress-daromaddan yuqori.")
    elif marja < 0.08:
        reasons.append("Foyda marjasi past (8% dan kam).")

    if dscr < 1.0:
        reasons.append("DSCR < 1.0: to‘lov qobiliyati yetarli emas (bankcha signal).")
    elif dscr < 1.2:
        reasons.append("DSCR 1.0–1.2: ehtiyotkor zona.")

    if qarz_n > 0.40:
        reasons.append("Qarz/daromad nisbati yuqori.")
    if yuk > 0.30:
        reasons.append("Kredit oylik to‘lovi yuklamasi yuqori (daromadga nisbatan).")

    if prob >= 0.55:
        reasons.append("ML model ehtimoli yuqori (statistik risk signali).")

    if not reasons:
        reasons.append("Katta salbiy signal topilmadi, ko‘rsatkichlar sog‘lom ko‘rinadi.")
    return reasons[:8]

def policy_decision(prob: float, stress_profit: float, dscr: float):
    """
    FINAL qaror (bank policy):
    - Hard stop: stress_profit <= 0 yoki DSCR < 1.0 => kredit yo‘q
    - Aks holda: DSCR + prob zonasi
    """
    if stress_profit <= 0:
        return 0, "🔴 RED", "❌ Stress-holatda sof foyda manfiy/0. Kredit tavsiya qilinmaydi (Hard rule)."
    if dscr < 1.0:
        return 0, "🔴 RED", "❌ DSCR < 1.0. Kredit tavsiya qilinmaydi (Hard rule)."
    if dscr < 1.2 or prob >= 0.25:
        return 1, "🟡 YELLOW", "🟡 Ehtiyotkor tavsiya: kichikroq kredit / uzunroq muddat / qo‘shimcha kafolat talab qilinishi mumkin."
    return 1, "🟢 GREEN", "🟢 Kredit tavsiya qilinadi: ko‘rsatkichlar yaxshi."

def plot_feature_importance(model, feature_names, top_n: int = 10):
    """Tree model bo‘lsa feature_importances_ chizadi."""
    if not hasattr(model, "feature_importances_"):
        return None

    importances = np.array(model.feature_importances_, dtype=float)
    fn = list(feature_names)

    m = min(len(importances), len(fn))
    importances = importances[:m]
    fn = fn[:m]

    idx = np.argsort(importances)[::-1][:top_n]
    top_feats = [fn[i] for i in idx]
    top_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(top_feats))
    ax.barh(y, top_vals)
    ax.set_yticks(y)
    ax.set_yticklabels(top_feats)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importance")
    plt.tight_layout()
    return fig

# =========================
# UI (Minimal only)
# =========================
st.subheader("🧾 Asosiy ma'lumotlar (Minimal)")

oylik_daromad = st.number_input("Oylik daromad (UZS)", min_value=0, value=50_000_000, step=1_000_000)
oylik_xarajat = st.number_input("Oylik xarajat (UZS)", min_value=0, value=35_000_000, step=1_000_000)
qarz_miqdori = st.number_input("Hozirgi kredit summasi / qarz (UZS)", min_value=0, value=10_000_000, step=500_000)
biznes_yoshi = st.number_input("Biznes yoshi (yil)", min_value=0, value=3, step=1)  # trainingda bo‘lsa ishlaydi
kech_tolovlar = st.number_input("Kechiktirilgan to‘lovlar soni", min_value=0, max_value=10, value=1, step=1)  # trainingda bo‘lsa ishlaydi
soliq_qarzi = st.number_input("Soliq qarzi (UZS)", min_value=0, value=0, step=500_000)  # trainingda bo‘lsa ishlaydi
mavsumiylik = st.selectbox("Mavsumiylik", [0, 1], format_func=lambda x: "Ha" if x == 1 else "Yo‘q")

st.markdown("#### 💳 Kredit shartlari (Avto hisoblash)")
annual_rate = st.slider("Yillik stavka (taxminiy)", 0.10, 0.60, 0.32, 0.01)
months = st.selectbox("Muddat (oy)", [12, 18, 24, 36], index=0)

auto_payment = estimate_monthly_payment(float(qarz_miqdori), float(annual_rate), int(months))
st.caption(f"📌 Avtomatik hisoblangan oylik to‘lov: **{int(auto_payment):,} UZS**")

st.divider()

# =========================
# Compute block
# =========================
if st.button("📊 Hisoblash", type="primary"):
    if oylik_daromad <= 0:
        st.error("Oylik daromad 0 bo‘lishi mumkin emas.")
        st.stop()

    # Stress-test
    stress_income = float(oylik_daromad) * (0.75 if int(mavsumiylik) == 1 else 1.0)
    stress_profit = stress_income - float(oylik_xarajat)
    dscr = (stress_profit / auto_payment) if auto_payment > 0 else float("inf")

    # Minimal user_data (Advanced yo‘q)
    user_data = {
        "oylik_daromad": float(oylik_daromad),
        "oylik_xarajat": float(oylik_xarajat),
        "qarz_miqdori": float(qarz_miqdori),
        "kredit_oylik_tolovi": float(auto_payment),
        "biznes_yoshi": int(biznes_yoshi),
        "kech_tolovlar": int(kech_tolovlar),
        "soliq_qarzi": float(soliq_qarzi),
        "mavsumiylik": int(mavsumiylik),
    }

    X_row = build_row(user_data)
    prob = safe_predict_proba(model, X_row)

    ml_pred = int(prob >= THRESHOLD)
    ok, policy_color, policy_msg = policy_decision(prob, stress_profit, dscr)

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ML bankrot ehtimoli", f"{prob*100:.1f}%")
    c2.metric("ML qarori", "Bankrot (1)" if ml_pred else "Barqaror (0)")
    c3.metric("ML risk darajasi", risk_badge(prob))
    c4.metric("DSCR (stress)", f"{dscr:.2f}")
    c5.metric("Stress foyda", f"{int(stress_profit):,} UZS")

    st.pyplot(draw_gauge(prob))

    # Policy decision
    st.subheader("🏦 Yakuniy qaror (Bank policy)")
    if ok == 1 and policy_color == "🟢 GREEN":
        st.success(policy_msg)
    elif ok == 1 and policy_color == "🟡 YELLOW":
        st.warning(policy_msg)
    else:
        st.error(policy_msg)

    # Reasons
    st.subheader("🧠 Tushuntirish (mantiqiy sabablar)")
    reasons = explain_reasons_minimal(
        stress_profit=stress_profit,
        dscr=dscr,
        mavsumiylik=int(mavsumiylik),
        prob=prob,
        daromad=float(oylik_daromad),
        xarajat=float(oylik_xarajat),
        qarz=float(qarz_miqdori),
        monthly_payment=float(auto_payment),
    )
    for r in reasons:
        st.write("•", r)

    # Loan suggestion (only if policy allows)
    st.subheader("💳 Kredit tavsiyasi")
    if ok == 0:
        st.info("Kredit oraliqlari hisoblanmadi, chunki policy bo‘yicha kredit tavsiya qilinmadi.")
    else:
        # GREEN: 30–45%, YELLOW: 20–35%
        if policy_color == "🟢 GREEN":
            pay_low, pay_high = 0.30, 0.45
        else:
            pay_low, pay_high = 0.20, 0.35

        safe_pay_low = max(stress_profit * pay_low, 0)
        safe_pay_high = max(stress_profit * pay_high, safe_pay_low)

        k_low = max_principal_from_payment(safe_pay_low, annual_rate, months)
        k_high = max_principal_from_payment(safe_pay_high, annual_rate, months)

        k_low = max(k_low, 1_000_000)
        k_high = max(k_high, k_low + 500_000)

        st.write(f"**Stress-holatda sof foyda:** {int(stress_profit):,} UZS")
        st.write(f"**Xavfsiz oyiga to‘lov oralig‘i:** {int(safe_pay_low):,} – {int(safe_pay_high):,} UZS")
        st.write(f"**Tavsiya qilinadigan kredit oralig‘i (annuitet):** {int(k_low):,} – {int(k_high):,} UZS")
        st.write(f"**Muddat:** {months} oy | **Yillik stavka:** {annual_rate*100:.1f}%")

    # Feature importance (optional, but kept)
    st.divider()
    st.subheader("🔎 Model Feature Importance (Top 10)")
    fig_imp = plot_feature_importance(model, FEATURES, top_n=10)
    if fig_imp is None:
        st.info("Bu modelda feature_importances_ mavjud emas (tree bo‘lmagan model bo‘lishi mumkin).")
    else:
        st.pyplot(fig_imp)

    st.caption(f"ML Threshold (reference): {THRESHOLD:.3f}")

else:
    st.info("Minimal ma’lumotlarni kiriting va **📊 Hisoblash** tugmasini bosing.")