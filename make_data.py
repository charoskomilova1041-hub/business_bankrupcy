from data_generator import generate_dataset

df = generate_dataset(n=10000, seed=42)
df.to_csv("uzbiz_dataset.csv", index=False, encoding="utf-8-sig")
print("Saved: uzbiz_dataset.csv", df.shape)