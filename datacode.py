import random
import pandas as pd

random.seed(42)

data = []

sohalar = ["Savdo", "Xizmat", "Ishlab chiqarish"]
hududlar = ["Toshkent", "Viloyat"]

N = 1000  # bizneslar soni

for _ in range(N):
    biznes_yoshi = random.randint(0, 15)
    sohasi = random.choice(sohalar)
    hudud = random.choice(hududlar)

    # Oylik daromad
    if sohasi == "Savdo":
        oylik_daromad = random.randint(20_000_000, 200_000_000)
    elif sohasi == "Xizmat":
        oylik_daromad = random.randint(10_000_000, 150_000_000)
    else:
        oylik_daromad = random.randint(50_000_000, 400_000_000)

    if hudud == "Toshkent":
        oylik_daromad = int(oylik_daromad * random.uniform(1.1, 1.4))

    xarajat_nisbati = random.uniform(0.6, 1.2)
    oylik_xarajat = int(oylik_daromad * xarajat_nisbati)

    # Shovqin
    oylik_daromad = int(oylik_daromad * random.uniform(0.9, 1.1))
    oylik_xarajat = int(oylik_xarajat * random.uniform(0.95, 1.15))

    sof_foyda = oylik_daromad - oylik_xarajat

    # Kredit miqdori
    if biznes_yoshi == 0:
        kredit_miqdori = random.choice([0, random.randint(20_000_000, 80_000_000)])
    elif biznes_yoshi <= 1:
        kredit_miqdori = random.randint(20_000_000, 100_000_000)
    elif biznes_yoshi <= 5:
        kredit_miqdori = random.randint(50_000_000, 300_000_000)
    else:
        kredit_miqdori = random.choice([0, random.randint(100_000_000, 500_000_000)])

    kredit_oylik_tolovi = int(kredit_miqdori * random.uniform(0.02, 0.05)) if kredit_miqdori > 0 else 0

    soliq_qarzi = random.choice([
        0,
        random.randint(1_000_000, 30_000_000),
        random.randint(30_000_000, 80_000_000)
    ])

    if sohasi == "Ishlab chiqarish":
        xodim_soni = random.randint(5, 100)
    else:
        xodim_soni = random.randint(1, 50)

    kech_tolovlar_soni = random.randint(0, 10)

    foyda_margini = sof_foyda / oylik_daromad if oylik_daromad > 0 else 0
    qarz_daromadga_nisbati = kredit_miqdori / oylik_daromad if oylik_daromad > 0 else 0
    xarajat_nisbati = oylik_xarajat / oylik_daromad if oylik_daromad > 0 else 0
    kredit_yuklama = kredit_oylik_tolovi / oylik_daromad if oylik_daromad > 0 else 0

    # Bankrotlik balli
    risk_ball = 0
    if sof_foyda < 0:
        risk_ball += 3
    if foyda_margini < 0.05:
        risk_ball += 1
    if kech_tolovlar_soni > 3:
        risk_ball += 2
    if soliq_qarzi > 15_000_000:
        risk_ball += 2
    if kredit_yuklama > 0.4:
        risk_ball += 2
    if biznes_yoshi < 2:
        risk_ball += 1
    if qarz_daromadga_nisbati > 1:
        risk_ball += 1

    bankrotlik_riski = 1 if risk_ball >= 5 else 0

    data.append([
        oylik_daromad,
        oylik_xarajat,
        sof_foyda,
        kredit_miqdori,
        kredit_oylik_tolovi,
        soliq_qarzi,
        xodim_soni,
        biznes_yoshi,
        kech_tolovlar_soni,
        sohasi,
        hudud,
        foyda_margini,
        qarz_daromadga_nisbati,
        xarajat_nisbati,
        kredit_yuklama,
        bankrotlik_riski
    ])

columns = [
    "oylik_daromad_uzs",
    "oylik_xarajat_uzs",
    "sof_foyda_uzs",
    "kredit_miqdori_uzs",
    "kredit_oylik_tolovi",
    "soliq_qarzi_uzs",
    "xodim_soni",
    "biznes_yoshi_yil",
    "kech_tolovlar_soni",
    "sohasi",
    "hudud",
    "foyda_margini",
    "qarz_daromadga_nisbati",
    "xarajat_nisbati",
    "kredit_yuklama",
    "bankrotlik_riski"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("dataset_uzbekcha.csv", index=False)

print("✅ Dataset yaratildi: dataset_uzbekcha.csv")
print(df.head())
print("\n📊 Bankrotlik balans:")
print(df["bankrotlik_riski"].value_counts(normalize=True))
