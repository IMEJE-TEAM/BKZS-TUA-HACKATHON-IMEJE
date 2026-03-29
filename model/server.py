import os
import asyncio
import json
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# TensorFlow'un gereksiz loglarını kapatalım
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

app = FastAPI(title="IMEJE-BKZS Anti-Spoofing Web Service")

# CORS izinleri (GitHub Pages üzerinden React/HTML dosyalarının bağlanmasına izin verir)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global değişkenler (Sunucu başlarken RAM'e alınacak)
SIM_DATA = []
TIME_STEPS = 30

def precompute(model, scaled, ts):
    N = len(scaled)
    nf = scaled.shape[1]
    probs = np.zeros(N, dtype=np.float32)
    nw = N - ts + 1
    if nw <= 0:
        return probs
    # Orijinal simulation.py dosyasındaki tahmin algoritması mantığı
    windows = np.lib.stride_tricks.sliding_window_view(scaled, (ts, nf))
    windows = windows.reshape(nw, ts, nf)
    preds = model.predict(windows, batch_size=4096, verbose=0).flatten()
    probs[ts - 1:] = preds
    return probs

@app.on_event("startup")
async def startup_event():
    print("Masaüstü simülasyonu Web'e aktarılıyor... Veriler yükleniyor...")
    
    # Gerçek yolları hesapla (Çalıştırıldığı yer ile model klasörü uyumu için)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "crnn_model.h5")
    scaler_path = os.path.join(base_dir, "crnn_scaler.pkl")
    data_path = os.path.join(base_dir, "test_senaryosu.csv")

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print("UYARI: Model veya Veri bulunamadı. Lütfen yolları kontrol edin.")
        return

    # Modeli ve Scaler'ı Yükle
    model = tf.keras.models.load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # CSV Verisini oku
    df = pd.read_csv(data_path)
    features = [c for c in df.columns if c != 'Label']
    
    # Tahminleri baştan tek seferde hesapla (Müthiş RAM ve İşlemci tasarrufu sağlar)
    print(f"Toplam {len(df)} satır veri için AI tahminleri önceden hesaplanıyor...")
    scaled = scaler.transform(df[features])
    probs = precompute(model, scaled, TIME_STEPS)
    
    # Frontend'e yollanacak veriyi JSON-friendly (liste içinde dict) formata dönüştür
    global SIM_DATA
    for i in range(len(df)):
        row_dict = {feat: float(df.iloc[i][feat]) for feat in features}
        row_dict['index'] = i
        row_dict['true_label'] = int(df.iloc[i].get('Label', 0))
        row_dict['probability'] = float(probs[i])
        row_dict['is_spoofed'] = bool(probs[i] >= 0.5)
        SIM_DATA.append(row_dict)
        
    print("Sistem Hazır! /ws WebSocket adresi üzerinden yayın yapmaya başlayabilir.")

@app.get("/")
def read_root():
    return {"status": "IMEJE-BKZS Anti-Spoofing Web Server (FastAPI) is Running!"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Yeni bir İstemci bağlandı (GitHub Pages)")
    
    # Saniyede 10 veri (0.1) gönderir. Masaüstündeki çok hızlı bir akış yerine Web için akıcı tutuldu.
    # Orijinali (0.033) = ~30 fps
    INTERVAL = 0.1  
    
    try:
        if not SIM_DATA:
            await websocket.send_json({"error": "Sunucuda veri yok veya AI modeli yüklenemedi."})
            return

        idx = 0
        total_rows = len(SIM_DATA)
        while True:
            # Döngü sonuna geldiğinde bitmesin diye sarmal çalışır
            if idx >= total_rows:
                idx = 0 
                
            chunk = SIM_DATA[idx:idx+5]  # Frontend grafikleri daha hızlı dolsun diye 5'er 5'er atabiliriz veya 1'erli.
            await websocket.send_json({"data": chunk})
            
            idx += 5
            await asyncio.sleep(INTERVAL)  
            
    except WebSocketDisconnect:
        print("İstemci bağlantısı koptu.")
    except Exception as e:
        print("Soket sırasında Hata:", e)

if __name__ == "__main__":
    import uvicorn
    # Sadece testi kendi bilgisayarınızda (Desktop) yaparken çalışır
    # Render bunu kullanmaz (Başlatma komutunuz uvicorn model.server:app'tir)
    uvicorn.run("server:app", host="127.0.0.0", port=8000, reload=True)
