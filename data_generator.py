# data_generator.py
import random
import math
import pandas as pd

SOHALAR = ["Savdo", "Xizmat", "Ishlab chiqarish"]
HUDUDLAR = ["Toshkent", "Viloyat"]

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))

def calculate_risk(row: dict) -> int:
    """
    0–100 oralig'ida risk_score.
    Moliyaviy + intizom + soliq + soha/hudud + import/valyuta + naqd/mavsumiylik + xodimlar.
    """
    risk = 0

    dar = row["oylik_daromad"]
    xar = row["oylik_xarajat"]
    qarz = row["qarz_miqdori"]
    tol = row["kredit_oylik_tolovi"]

    sof_foyda = dar - xar
    foyda_marjasi = sof_foyda / max(dar, 1)
    xarajat_nisbati = xar / max(dar, 1)
    qarz_nisbati = qarz / max(dar, 1)
    kredit_yuklama = tol / max(dar, 1)

    # 1) Moliyaviy holat
    if sof_foyda < 0:
        risk += 30
    elif sof_foyda < 5_000_000:
        risk += 18

    if foyda_marjasi < 0.03:
        risk += 18
    elif foyda_marjasi < 0.08:
        risk += 10

    # Xarajat juda yuqori bo'lsa
    if xarajat_nisbati > 0.95:
        risk += 18
    elif xarajat_nisbati > 0.85:
        risk += 10

    # Qarz/daromad
    if qarz_nisbati > 0.60:
        risk += 20
    elif qarz_nisbati > 0.40:
        risk += 12

    # Kredit to'lovi yuklamasi
    if kredit_yuklama > 0.45:
        risk += 18
    elif kredit_yuklama > 0.30:
        risk += 10

    # 2) Biznes yoshi
    if row["biznes_yoshi"] < 2:
        risk += 12
    elif row["biznes_yoshi"] < 5:
        risk += 6

    # 3) To'lov intizomi
    kt = row["kech_tolovlar"]
    if kt >= 5:
        risk += 18
    elif kt >= 2:
        risk += 10
    elif kt == 1:
        risk += 5

    # 4) Soliq qarzi
    sq = row["soliq_qarzi"]
    if sq > 20_000_000:
        risk += 12
    elif sq > 5_000_000:
        risk += 6

    # 5) Soha / hudud risklari
    if row["soha"] == "Savdo":
        risk += 6
    elif row["soha"] == "Ishlab chiqarish":
        risk += 4

    if row["hudud"] == "Viloyat":
        risk += 5

    # 6) Naqd ulushi
    naqd = row["naqd_ulushi"]
    if naqd < 0.40:
        risk += 10
    elif naqd < 0.55:
        risk += 5

    # 7) Mavsumiylik
    if row["mavsumiylik"] == 1 and foyda_marjasi < 0.08:
        risk += 6

    # 8) Import + valyuta (✅ monotonic va sezilarli)
    if row["import_bogliq"] == 1:
        v = row["valyuta_kurs_xavfi"]
        risk += v * 15  # 0.2->+3, 0.8->+12, 1.0->+15

    # 9) Xodimlar
    xodim = row["xodimlar_soni"]
    if xodim <= 2:
        risk += 6
    elif xodim > 25:
        risk -= 4

    return int(min(max(risk, 0), 100))

def generate_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Mukammal generator:
    - Soha/Hudud daromad diapazoni + koeffitsient
    - Xarajat ba'zan daromaddan katta (zarar)
    - Qarz va oylik to'lov bog'langan (✅ yaxshilandi)
    - Import soha bo'yicha ehtimollik bilan
    - Valyuta xavfi importga bog'liq taqsimlanadi
    - Bankrot label: risk + shovqin -> sigmoid (✅ bankrot ulushi sozlandi)
    """
    random.seed(seed)
    data = []

    for _ in range(n):
        soha = random.choice(SOHALAR)
        hudud = random.choice(HUDUDLAR)

        # 1) Daromad: soha bo'yicha realistik diapazon
        if soha == "Savdo":
            dar = random.randint(10_000_000, 250_000_000)
        elif soha == "Xizmat":
            dar = random.randint(5_000_000, 180_000_000)
        else:
            dar = random.randint(20_000_000, 400_000_000)

        # 2) Hudud koeffitsienti
        if hudud == "Toshkent":
            dar = int(dar * random.uniform(1.05, 1.35))
        else:
            dar = int(dar * random.uniform(0.85, 1.10))

        # 3) Xarajat (ba'zan zarar)
        xar = int(dar * random.uniform(0.55, 1.05))

        # 4) Qarz (0–70% daromad)
        qarz = int(dar * random.uniform(0.0, 0.70))

        # 5) Kredit oylik to'lovi qarzga bog'liq (✅ kuchaytirildi)
        tol = int((qarz / 12) * random.uniform(0.10, 0.25))

        # 6) Import ehtimoli soha bo'yicha
        if soha == "Savdo":
            import_bogliq = 1 if random.random() < 0.65 else 0
        elif soha == "Ishlab chiqarish":
            import_bogliq = 1 if random.random() < 0.45 else 0
        else:
            import_bogliq = 1 if random.random() < 0.30 else 0

        # 7) Valyuta xavfi importga bog'liq taqsimlanadi
        if import_bogliq == 1:
            valyuta = round(random.uniform(0.25, 1.0), 2)
        else:
            valyuta = round(random.uniform(0.0, 0.55), 2)

        # 8) Naqd ulushi va mavsumiylik
        naqd = round(random.uniform(0.25, 0.95), 2)
        mavsumiy = random.choice([0, 1])

        # 9) Biznes yoshi, intizom, soliq, xodim
        biznes_yoshi = random.randint(0, 15)
        kech_tolovlar = random.randint(0, 10)
        soliq_qarzi = random.choice([0, random.randint(1_000_000, 60_000_000)])
        xodimlar_soni = random.randint(1, 120)

        row = {
            "oylik_daromad": dar,
            "oylik_xarajat": xar,
            "qarz_miqdori": qarz,
            "kredit_oylik_tolovi": tol,
            "biznes_yoshi": biznes_yoshi,
            "kech_tolovlar": kech_tolovlar,
            "soliq_qarzi": soliq_qarzi,
            "soha": soha,
            "hudud": hudud,
            "naqd_ulushi": naqd,
            "mavsumiylik": mavsumiy,
            "xodimlar_soni": xodimlar_soni,
            "import_bogliq": import_bogliq,
            "valyuta_kurs_xavfi": valyuta,
        }

        # 10) Risk score
        risk = calculate_risk(row)

        # 11) Bankrot label (✅ bankrot ulushini tushirdik: -55 -> -60)
        noise = random.gauss(0, 6)
        z = (risk + noise - 60) / 8
        p_bankrot = sigmoid(z)

        row["risk_score"] = risk
        row["bankrot"] = 1 if random.random() < p_bankrot else 0

        data.append(row)

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_dataset(n=5000, seed=42)

    print(df.head())

    # Tez tekshiruvlar
    df["foyda"] = df["oylik_daromad"] - df["oylik_xarajat"]
    print("\nBankrot ulushi:", round(df["bankrot"].mean(), 3))
    print("Zarar (foyda<0) ulushi:", round((df["foyda"] < 0).mean(), 3))
    print("\nImport bo'yicha valyuta o'rtacha:")
    print(df.groupby("import_bogliq")["valyuta_kurs_xavfi"].mean().round(3))