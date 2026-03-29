#!/usr/bin/env python3
"""
IMEJE-BKZS Anti-Spoofing Web Simulasyon Baslatici

Kullanim:
    python start_website.py

Bu script:
    1. Eger simulation_data.json yoksa export_data.py'yi calistirir
    2. Yerel HTTP sunucusu baslatir
    3. Tarayiciyi acar
"""
import os
import sys
import subprocess
import webbrowser
import http.server
import socketserver

ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
WEBSITE_DIR = os.path.join(ROOT_DIR, "website")
DATA_FILE   = os.path.join(WEBSITE_DIR, "data", "simulation_data.json")
MODEL_DIR   = os.path.join(ROOT_DIR, "model")
EXPORT_SCRIPT = os.path.join(MODEL_DIR, "export_data.py")

PORT = 8000


def main():
    print("=" * 50)
    print("  IMEJE-BKZS Anti-Spoofing Web Simulasyon")
    print("=" * 50)

    # 1. Veri dosyasi kontrolu
    if not os.path.exists(DATA_FILE):
        print("\n  Simulasyon verisi bulunamadi.")
        print("  export_data.py calistiriliyor...\n")

        if not os.path.exists(EXPORT_SCRIPT):
            print(f"  HATA: {EXPORT_SCRIPT} bulunamadi!")
            sys.exit(1)

        result = subprocess.run(
            [sys.executable, EXPORT_SCRIPT],
            cwd=MODEL_DIR
        )

        if result.returncode != 0:
            print("\n  HATA: Veri export islemi basarisiz!")
            sys.exit(1)

        if not os.path.exists(DATA_FILE):
            print(f"\n  HATA: {DATA_FILE} olusturulamadi!")
            sys.exit(1)

    size_mb = os.path.getsize(DATA_FILE) / (1024 * 1024)
    print(f"\n  Veri dosyasi: {size_mb:.1f} MB")

    # 2. HTTP sunucusu baslat
    os.chdir(WEBSITE_DIR)

    handler = http.server.SimpleHTTPRequestHandler
    handler.log_message = lambda *args: None  # Sessiz mod

    url = f"http://localhost:{PORT}"
    print(f"  Sunucu: {url}")
    print(f"  Durdurmak icin: Ctrl+C\n")

    # 3. Tarayici ac
    webbrowser.open(url)

    with socketserver.TCPServer(("", PORT), handler) as httpd:
        httpd.allow_reuse_address = True
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Sunucu durduruldu.")


if __name__ == "__main__":
    main()
