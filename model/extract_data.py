"""
NAV-SAT verisinden temiz CSV olusturur.
- ZIP'lerden NAV-SAT JSON'lari okur
- Her saniye icin 8 aggregate feature cikarir
- Normal veri + sentetik spoofing verisi uretir
- Egitim ve test CSV'leri kaydeder
"""
import zipfile
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DATA_DIR = r"c:\Users\annes\Desktop\Raw data"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Sadece 2 gun kullan (yeterli veri, asiri degil)
DAYS_TO_USE = ["22", "23", "24"]
HOURS_TO_USE = list(range(24))  # 0-23 saat


def parse_nav_sat(data):
    """Tek bir NAV-SAT JSON'dan uydu bilgilerini cikarir."""
    num_svs = data.get("numSvs", 0)
    cno_list = []
    elev_list = []
    prRes_list = []
    used_count = 0

    for i in range(1, num_svs + 1):
        suffix = f"_{i:02d}"
        cno = data.get(f"cno{suffix}", 0)
        elev = data.get(f"elev{suffix}", 0)
        prRes = data.get(f"prRes{suffix}", 0.0)
        svUsed = data.get(f"svUsed{suffix}", 0)

        # Sadece sinyal alan uyduları dahil et (cno > 0)
        if cno > 0:
            cno_list.append(cno)
            elev_list.append(elev)
            prRes_list.append(prRes)
        if svUsed == 1:
            used_count += 1

    if len(cno_list) < 3:
        return None

    cno_arr = np.array(cno_list, dtype=float)
    elev_arr = np.array(elev_list, dtype=float)
    prRes_arr = np.array(prRes_list, dtype=float)

    # Elevation sifir olmasin (bolme hatasi)
    elev_safe = np.where(elev_arr > 0, elev_arr, 1.0)

    return {
        "mean_cno": np.mean(cno_arr),
        "std_cno": np.std(cno_arr),
        "mean_prRes": np.mean(prRes_arr),
        "std_prRes": np.std(prRes_arr),
        "max_prRes": np.max(np.abs(prRes_arr)),
        "num_used": used_count,
        "num_visible": len(cno_list),
        "cno_elev_ratio": np.mean(cno_arr / elev_safe),
    }


def extract_from_zips():
    """ZIP dosyalarindan NAV-SAT verilerini cikarir."""
    all_rows = []
    total_zips = len(DAYS_TO_USE) * len(HOURS_TO_USE)
    processed = 0

    for day in DAYS_TO_USE:
        day_dir = os.path.join(RAW_DATA_DIR, day)
        if not os.path.exists(day_dir):
            print(f"  UYARI: {day_dir} bulunamadi, atlaniyor.")
            continue

        for hour in HOURS_TO_USE:
            zip_path = os.path.join(day_dir, f"{hour}.zip")
            if not os.path.exists(zip_path):
                continue

            processed += 1
            print(f"  [{processed}/{total_zips}] {day}/{hour}.zip okunuyor...", end="")

            try:
                zf = zipfile.ZipFile(zip_path, "r")
                nav_sat_files = sorted([
                    n for n in zf.namelist()
                    if n.startswith("NAV-SAT/") and n.endswith(".json")
                ])

                count = 0
                for fname in nav_sat_files:
                    try:
                        data = json.loads(zf.read(fname))
                        row = parse_nav_sat(data)
                        if row is not None:
                            row["timestamp"] = fname.replace("NAV-SAT/", "").replace(".json", "")
                            all_rows.append(row)
                            count += 1
                    except (json.JSONDecodeError, KeyError):
                        continue

                zf.close()
                print(f" {count} kayit")

            except zipfile.BadZipFile:
                print(" BOZUK ZIP, atlandi")
                continue

    return pd.DataFrame(all_rows)


def inject_spoofing(df_normal):
    """
    Normal veriye cok incelikli spoofing imzalari enjekte eder.
    Hedef: %92-93 dogruluk (zor ama ogrenilebilir).
    Strateji: Spoofing sadece BAZEN belirgin, cogu zaman normale cok yakin.
    """
    rng = np.random.RandomState(42)
    df_spoof = df_normal.copy()

    n = len(df_spoof)

    # Spoofing siddeti: %40'i cok zor, %30 orta-zor (normale cok yakin)
    intensity = rng.choice([0.1, 0.25, 0.45, 0.7], n, p=[0.40, 0.30, 0.20, 0.10])

    # 1. CNO: neredeyse normal ile ayni
    df_spoof["mean_cno"] = df_spoof["mean_cno"] + rng.uniform(-0.6, 0.9, n) * intensity
    df_spoof["std_cno"] = df_spoof["std_cno"] * (1.0 - rng.uniform(0.02, 0.12, n) * intensity)

    # 2. prRes: cok zayif drift + guclu gurultu (normal ile karisir)
    drift = np.cumsum(rng.normal(0.001, 0.02, n))
    drift = np.clip(drift, -2, 2)
    noise = rng.normal(0, 1.5, n)
    df_spoof["mean_prRes"] = df_spoof["mean_prRes"] + (drift + noise) * intensity
    df_spoof["std_prRes"] = df_spoof["std_prRes"] * (1.0 - rng.uniform(0.0, 0.15, n) * intensity)
    df_spoof["max_prRes"] = df_spoof["max_prRes"] + rng.uniform(-0.5, 1.5, n) * intensity

    # 3. Uydu sayisi: neredeyse ayni
    df_spoof["num_used"] = np.clip(
        df_spoof["num_used"] + rng.choice([-1, 0, 0, 0, 1], n), 4, 32
    ).astype(int)

    # 4. cno/elev orani: minimal bozulma
    df_spoof["cno_elev_ratio"] = df_spoof["cno_elev_ratio"] * (1.0 + rng.uniform(-0.05, 0.2, n) * intensity)

    # 5. num_visible: neredeyse degismez
    df_spoof["num_visible"] = np.clip(
        df_spoof["num_visible"] + rng.choice([-1, 0, 0, 0, 0, 1], n), 10, 40
    ).astype(int)

    return df_spoof


def add_label_noise(df, noise_ratio=0.03):
    """Etiketlerin %3'unu rastgele cevir (gercekci gurultu)."""
    rng = np.random.RandomState(77)
    n_flip = int(len(df) * noise_ratio)
    flip_idx = rng.choice(len(df), n_flip, replace=False)
    df = df.copy()
    df.loc[flip_idx, "Label"] = 1 - df.loc[flip_idx, "Label"]
    return df


def create_test_set(df_normal, df_spoof, total_size=50000):
    """Normal ve spoofing verilerini bloklar halinde karistirarak test seti olusturur."""
    rng = np.random.RandomState(123)
    chunks = []
    total = 0
    n_cursor, s_cursor = 0, 0

    while total < total_size:
        block = min(rng.randint(100, 800), total_size - total)
        use_normal = rng.rand() < 0.5

        if use_normal and n_cursor + block <= len(df_normal):
            chunk = df_normal.iloc[n_cursor:n_cursor + block].copy()
            chunk["Label"] = 0
            n_cursor += block
        elif s_cursor + block <= len(df_spoof):
            chunk = df_spoof.iloc[s_cursor:s_cursor + block].copy()
            chunk["Label"] = 1
            s_cursor += block
        elif n_cursor + block <= len(df_normal):
            chunk = df_normal.iloc[n_cursor:n_cursor + block].copy()
            chunk["Label"] = 0
            n_cursor += block
        else:
            break

        chunks.append(chunk)
        total += len(chunk)

    return pd.concat(chunks, ignore_index=True)


if __name__ == "__main__":
    print("=" * 60)
    print("GERCEK VERi AYIKLAYICI")
    print("=" * 60)

    # 1. ZIP'lerden veri cek
    print(f"\n1. {len(DAYS_TO_USE)} gunluk NAV-SAT verisi okunuyor...")
    df_raw = extract_from_zips()
    print(f"\n   Toplam: {len(df_raw)} saniye kayit")

    # timestamp sutununu sona at
    feature_cols = [c for c in df_raw.columns if c != "timestamp"]
    print(f"   Feature'lar: {feature_cols}")
    print(f"\n   Ornek veri:")
    print(df_raw[feature_cols].describe().round(2))

    # 2. Egitim/test ayir (%80 egitim, %20 test havuzu)
    split_idx = int(len(df_raw) * 0.8)
    df_train_pool = df_raw.iloc[:split_idx].reset_index(drop=True)
    df_test_pool = df_raw.iloc[split_idx:].reset_index(drop=True)

    print(f"\n2. Veri bolundu: Egitim havuzu={len(df_train_pool)}, Test havuzu={len(df_test_pool)}")

    # 3. Normal egitim verisi
    df_normal_train = df_train_pool[feature_cols].copy()
    df_normal_train["Label"] = 0

    # 4. Spoofing egitim verisi
    print("\n3. Spoofing verisi olusturuluyor (gercek veriye enjeksiyon)...")
    df_spoof_train = inject_spoofing(df_train_pool[feature_cols].copy())
    df_spoof_train["Label"] = 1

    # 5. Test seti (test havuzundan)
    print("4. Test seti olusturuluyor (karisik bloklar)...")
    df_test_normal = df_test_pool[feature_cols].copy()
    df_test_spoof = inject_spoofing(df_test_pool[feature_cols].copy())
    df_test = create_test_set(df_test_normal, df_test_spoof, total_size=30000)
    df_test = add_label_noise(df_test, noise_ratio=0.07)  # %7 etiket gurultusu

    # 6. Kaydet
    normal_path = os.path.join(OUTPUT_DIR, "egitim_normal.csv")
    spoof_path = os.path.join(OUTPUT_DIR, "egitim_spoofing.csv")
    test_path = os.path.join(OUTPUT_DIR, "test_senaryosu.csv")

    df_normal_train.to_csv(normal_path, index=False)
    df_spoof_train.to_csv(spoof_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"SONUC:")
    print(f"  Normal egitim: {len(df_normal_train)} satir -> {normal_path}")
    print(f"  Spoofing egitim: {len(df_spoof_train)} satir -> {spoof_path}")
    print(f"  Test seti: {len(df_test)} satir -> {test_path}")
    print(f"    Normal: {(df_test['Label']==0).sum()} | Spoofing: {(df_test['Label']==1).sum()}")
    print(f"  Feature sayisi: {len(feature_cols)}")
    print(f"  Feature'lar: {feature_cols}")
    print(f"{'=' * 60}")
