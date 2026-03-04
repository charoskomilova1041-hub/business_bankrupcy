import pandas as pd
from data_generator import generate_dataset

df = generate_dataset(n=5000)

# ------------------------------
# 1️⃣ Asosiy info
# ------------------------------
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# ------------------------------
# 2️⃣ Bankrot balans
# ------------------------------
print("\n=== BANKROT BALANS ===")
print(df["bankrot"].value_counts())
print(df["bankrot"].value_counts(normalize=True).round(3))

# ------------------------------
# 3️⃣ Risk score
# ------------------------------
print("\n=== RISK SCORE ===")
print(df["risk_score"].describe())

# ------------------------------
# 4️⃣ Zarar ulushi
# ------------------------------
df["foyda"] = df["oylik_daromad"] - df["oylik_xarajat"]
print("\nZarar ulushi (foyda<0):", round((df["foyda"] < 0).mean(), 3))

# ------------------------------
# 5️⃣ Import vs Valyuta
# ------------------------------
print("\n=== IMPORT vs VALYUTA ===")
print(df.groupby("import_bogliq")["valyuta_kurs_xavfi"].mean().round(3))

# ------------------------------
# 6️⃣ Valyuta riskga ta’siri
# ------------------------------
df["valyuta_bin"] = pd.cut(df["valyuta_kurs_xavfi"], bins=[0,0.2,0.4,0.6,0.8,1.0])

pivot = df.pivot_table(
    values="risk_score",
    index="valyuta_bin",
    columns="import_bogliq",
    aggfunc="mean"
)

print("\n=== VALYUTA → RISK TA'SIRI ===")
print(pivot.round(2))