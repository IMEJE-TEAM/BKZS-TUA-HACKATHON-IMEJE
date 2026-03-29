#!/usr/bin/env python3
"""
Simulation verisini web frontend icin JSON'a export eder.

Kullanim:
    cd model
    python export_data.py

Cikti:
    ../website/data/simulation_data.json
"""
import os, sys, warnings, json
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import pickle

TIME_STEPS     = 30
MODEL_PATH     = "crnn_model.h5"
SCALER_PATH    = "crnn_scaler.pkl"
TEST_DATA_PATH = "test_senaryosu.csv"

FEAT_CFG = [
    ('mean_prRes',     'Ort. Pseudorange Hatasi',  '#f0883e', 'm',    'Uydu mesafe olcum sapmasi'),
    ('std_prRes',      'Std Pseudorange Hatasi',   '#ef5350', 'm',    'Hata dagilim genisligi'),
    ('max_prRes',      'Maks Pseudorange Hatasi',  '#bc8cff', 'm',    'En buyuk anlik sapma'),
    ('mean_cno',       'Ort. Sinyal Gucu (C/N0)',  '#39d2c0', 'dBHz', 'Uydu sinyal kalitesi'),
    ('std_cno',        'Std Sinyal Gucu',          '#66bb6a', '',     'Sinyal kararsizligi'),
    ('cno_elev_ratio', 'Sinyal/Yukseklik Orani',   '#f778ba', '',     'Aci-normalize sinyal'),
]


def main():
    # Dosya kontrolleri
    for p in [MODEL_PATH, SCALER_PATH, TEST_DATA_PATH]:
        if not os.path.exists(p):
            print(f"HATA: '{p}' bulunamadi! Once train_model.py calistirin.")
            sys.exit(1)

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    print("Model yukleniyor...")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    print("Test verisi okunuyor...")
    df = pd.read_csv(TEST_DATA_PATH)
    features = [c for c in df.columns if c != 'Label']

    # Tahminleri hesapla
    print("Tahminler hesaplaniyor...")
    scaled = scaler.transform(df[features])
    N = len(scaled)
    nf = scaled.shape[1]
    probs = np.zeros(N, dtype=np.float32)

    nw = N - TIME_STEPS + 1
    if nw > 0:
        windows = np.lib.stride_tricks.sliding_window_view(scaled, (TIME_STEPS, nf))
        windows = windows.reshape(nw, TIME_STEPS, nf)
        preds = model.predict(windows, batch_size=4096, verbose=0).flatten()
        probs[TIME_STEPS - 1:] = preds

    print(f"  {N:,} tahmin hazir.")

    # JSON ciktisi olustur
    output = {
        "features": [],
        "labels": [int(x) for x in df['Label'].values],
        "probs": [round(float(x), 4) for x in probs],
        "data": {},
        "totalRows": N
    }

    for name, label, color, unit, desc in FEAT_CFG:
        if name in features:
            output["features"].append({
                "key": name,
                "label": label,
                "color": color,
                "unit": unit,
                "desc": desc
            })
            output["data"][name] = [round(float(x), 3) for x in df[name].values]

    # Kaydet
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "website", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "simulation_data.json")

    with open(out_path, 'w') as f:
        json.dump(output, f, separators=(',', ':'))

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nExport tamamlandi!")
    print(f"  Dosya: {out_path}")
    print(f"  Satir: {N:,}")
    print(f"  Boyut: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
