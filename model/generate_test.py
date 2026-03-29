import pandas as pd
import numpy as np

print("Test senaryosu uretici baslatiliyor...")
print("-" * 50)

# Egitim verilerini oku
df_normal = pd.read_csv("egitim_normal.csv")
df_spoof = pd.read_csv("egitim_spoofing.csv")

print(f"Normal veri: {len(df_normal)} satir")
print(f"Spoofing veri: {len(df_spoof)} satir")

# Son %20'lik dilimi test havuzu olarak ayir (egitimle karismasin)
normal_pool = df_normal.iloc[int(len(df_normal) * 0.8):].reset_index(drop=True)
spoof_pool = df_spoof.iloc[int(len(df_spoof) * 0.8):].reset_index(drop=True)

print(f"Normal havuz: {len(normal_pool)} satir")
print(f"Spoofing havuz: {len(spoof_pool)} satir")

# Rastgele bloklar halinde birlestir
rng = np.random.RandomState(42)
chunks = []
n_cursor = 0
s_cursor = 0
total_target = 50000  # toplam test satiri
total_added = 0

while total_added < total_target:
    # Blok boyutu: 50 - 500 arasi rastgele
    block_size = rng.randint(50, 500)
    block_size = min(block_size, total_target - total_added)

    # %50 normal, %50 spoofing secimi
    pick_normal = rng.rand() < 0.5

    if pick_normal and n_cursor + block_size <= len(normal_pool):
        chunk = normal_pool.iloc[n_cursor:n_cursor + block_size].copy()
        n_cursor += block_size
    elif s_cursor + block_size <= len(spoof_pool):
        chunk = spoof_pool.iloc[s_cursor:s_cursor + block_size].copy()
        s_cursor += block_size
    elif n_cursor + block_size <= len(normal_pool):
        chunk = normal_pool.iloc[n_cursor:n_cursor + block_size].copy()
        n_cursor += block_size
    else:
        break

    chunks.append(chunk)
    total_added += len(chunk)

df_test = pd.concat(chunks, ignore_index=True)

# Label sutununu int yap
df_test['Label'] = df_test['Label'].astype(int)

# Kaydet
df_test.to_csv("test_senaryosu.csv", index=False)

# Ozet
n_count = (df_test['Label'] == 0).sum()
s_count = (df_test['Label'] == 1).sum()
print(f"\nTest senaryosu olusturuldu: {len(df_test)} satir")
print(f"  Normal bloklar: {n_count} satir ({n_count/len(df_test)*100:.1f}%)")
print(f"  Spoofing bloklar: {s_count} satir ({s_count/len(df_test)*100:.1f}%)")
print(f"\nSutunlar: {list(df_test.columns)}")
print(df_test.head(10))
print("...")
print(df_test.tail(10))
print(f"\n'test_senaryosu.csv' kaydedildi.")
