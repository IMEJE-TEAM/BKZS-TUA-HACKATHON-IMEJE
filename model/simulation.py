#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMEJE-BKZS Anti-Spoofing — Canli Simulasyon v4

Kontroller:
    SPACE                 Duraklat / Devam
    HOME                  Canli uca don
"""

import os, sys, warnings, argparse
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import pickle

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.image import AxesImage

# ==============================================================================
TIME_STEPS     = 30
MODEL_PATH     = "crnn_model.h5"
SCALER_PATH    = "crnn_scaler.pkl"
TEST_DATA_PATH = "test_senaryosu.csv"

VIEW_WINDOW    = 800
BATCH_SIZE     = 5
INTERVAL_MS    = 33        # ~30 fps

# Tema
BG             = '#080c14'
PANEL          = '#0d1117'
PANEL2         = '#111922'
GRID           = '#171f2b'
TXT            = '#7b8794'
TXT_HI         = '#c9d1d9'
WHITE          = '#e6edf3'
GREEN          = '#26a69a'
RED            = '#ef5350'
RED_BG         = '#2a0a0a'
AMBER          = '#d29922'
PURPLE         = '#bc8cff'

FEAT_CFG = [
    ('mean_prRes',     'Ort. Pseudorange Hatasi',  '#f0883e', 'm',    'Uydu mesafe olcum sapmasi'),
    ('std_prRes',      'Std Pseudorange Hatasi',   '#ef5350', 'm',    'Hata dagilim genisligi'),
    ('max_prRes',      'Maks Pseudorange Hatasi',  '#bc8cff', 'm',    'En buyuk anlik sapma'),
    ('mean_cno',       'Ort. Sinyal Gucu (C/N0)',  '#39d2c0', 'dBHz', 'Uydu sinyal kalitesi'),
    ('std_cno',        'Std Sinyal Gucu',          '#66bb6a', '',     'Sinyal kararsizligi'),
    ('cno_elev_ratio', 'Sinyal/Yukseklik Orani',   '#f778ba', '',     'Aci-normalize sinyal'),
]


def load_all():
    for p in [MODEL_PATH, SCALER_PATH, TEST_DATA_PATH]:
        if not os.path.exists(p):
            print(f"HATA: '{p}' bulunamadi! Once train_model.py calistirin.")
            sys.exit(1)

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    df = pd.read_csv(TEST_DATA_PATH)
    features = [c for c in df.columns if c != 'Label']
    return model, scaler, df, features


def precompute(model, scaled, ts):
    N = len(scaled)
    nf = scaled.shape[1]
    probs = np.zeros(N, dtype=np.float32)
    nw = N - ts + 1
    if nw <= 0:
        return probs
    # sliding_window_view on 2D gives 4D: (nw, 1, ts, nf)
    windows = np.lib.stride_tricks.sliding_window_view(scaled, (ts, nf))
    windows = windows.reshape(nw, ts, nf)
    preds = model.predict(windows, batch_size=4096, verbose=0).flatten()
    probs[ts - 1:] = preds
    return probs


def build_spoof_bg(labels, N):
    """Spoofing bolgeleri icin 1-satir RGBA resim olustur. Her piksel 1 satir."""
    img = np.zeros((1, N, 4), dtype=np.float32)
    # Spoofing piksellerini kirmizi yap
    mask = labels[:N] == 1
    img[0, mask, 0] = 1.0   # R
    img[0, mask, 3] = 0.12  # Alpha
    return img


# ==============================================================================
class LiveSim:

    def __init__(self, model, scaler, df, features,
                 batch=BATCH_SIZE, interval=INTERVAL_MS, window=VIEW_WINDOW):
        self.df       = df
        self.features = features
        self.batch    = batch
        self.interval = interval
        self.window   = window
        self.N        = len(df)

        self.labels = df['Label'].values
        self.disp   = [(f, l, c, u, d) for f, l, c, u, d in FEAT_CFG if f in features]

        # Onceden hesapla
        print("  Tahminler hesaplaniyor...")
        scaled = scaler.transform(df[features])
        self.probs = precompute(model, scaled, TIME_STEPS)
        print(f"  {self.N:,} tahmin hazir.")

        # Ham degerler numpy
        self.raw = {f: df[f].values.astype(np.float64) for f, *_ in self.disp}

        # Spoofing arka plan resmi
        self.spoof_img = build_spoof_bg(self.labels, self.N)

        # Durum
        self.cursor    = 0
        self.paused    = False
        self.tp = self.fp = self.fn = self.tn = 0

        self._build()

    def _build(self):
        nf = len(self.disp)
        hrs = [0.4] + [1.0]*nf + [1.3]

        self.fig = plt.figure(figsize=(22, 2.0*nf + 4.5), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title('IMEJE-BKZS Anti-Spoofing')
        except:
            pass

        gs = gridspec.GridSpec(len(hrs), 1, height_ratios=hrs,
                               hspace=0.04, left=0.11, right=0.925, top=0.955, bottom=0.06)

        # ---- Baslik ----
        ax_h = self.fig.add_subplot(gs[0])
        ax_h.set_facecolor(BG)
        ax_h.axis('off')
        ax_h.text(0.0, 0.6, 'IMEJE-BKZS', fontsize=17, fontweight='bold',
                  color=WHITE, transform=ax_h.transAxes, fontfamily='monospace')
        ax_h.text(0.093, 0.6, 'GNSS Anti-Spoofing Canli Izleme', fontsize=15,
                  color=TXT, transform=ax_h.transAxes, fontfamily='monospace')

        self.txt_st = ax_h.text(1.0, 0.6, '', fontsize=8.5, color=TXT,
                                 transform=ax_h.transAxes, ha='right', fontfamily='monospace')
        self.txt_al = ax_h.text(0.5, -0.05, '', fontsize=14, fontweight='bold',
                                 color=RED, transform=ax_h.transAxes, ha='center',
                                 fontfamily='monospace',
                                 bbox=dict(boxstyle='round,pad=0.5', fc=RED_BG, ec=RED, alpha=0.0))
        self.ax_h = ax_h

        # ---- Feature panelleri ----
        self.panels = []
        self._spoof_imgs = []     # Her paneldeki AxesImage nesnesi

        for i, (feat, label, color, unit, desc) in enumerate(self.disp):
            ax = self.fig.add_subplot(gs[i + 1])
            ax.set_facecolor(PANEL if i % 2 == 0 else PANEL2)

            # Cizgiler (normal + kirmizi spoofing overlay)
            ln_norm,  = ax.plot([], [], color=color, lw=1.0, alpha=0.90, zorder=3)
            ln_spoof, = ax.plot([], [], color=RED, lw=1.4, alpha=0.92, zorder=4)

            u_s = f' ({unit})' if unit else ''
            lbl_txt = f'{label}{u_s}'
            ax.text(0.005, 0.92, lbl_txt, fontsize=7, color=color,
                    fontfamily='monospace', fontweight='bold',
                    transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.25', fc=BG, ec=color, alpha=0.75, lw=0.6))

            txt_v = ax.text(1.008, 0.5, '', fontsize=8, color=color, fontweight='bold',
                            fontfamily='monospace', transform=ax.transAxes,
                            ha='left', va='center',
                            bbox=dict(boxstyle='round,pad=0.2', fc=BG, ec=color, alpha=0.85, lw=0.7))

            ax.tick_params(colors=TXT, labelsize=6, length=2)
            ax.grid(True, color=GRID, alpha=0.5, lw=0.4)
            ax.set_xticklabels([])
            for sp in ax.spines.values():
                sp.set_color(GRID)

            self.panels.append({
                'feat': feat, 'ax': ax, 'color': color,
                'ln_norm': ln_norm, 'ln_spoof': ln_spoof,
                'txt_v': txt_v,
            })

        # ---- Model paneli ----
        self.ax_m = self.fig.add_subplot(gs[nf + 1])
        self.ax_m.set_facecolor(PANEL)

        self.ln_p,  = self.ax_m.plot([], [], color=PURPLE, lw=1.2, alpha=0.9, zorder=3)
        self.ln_pg, = self.ax_m.plot([], [], color=PURPLE, lw=4, alpha=0.08, zorder=2)
        self.ax_m.axhline(y=0.5, color=AMBER, ls='--', lw=0.9, alpha=0.5)

        self.txt_pv = self.ax_m.text(
            1.008, 0.75, '', fontsize=9, color=PURPLE, fontweight='bold',
            fontfamily='monospace', transform=self.ax_m.transAxes, ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc=BG, ec=PURPLE, alpha=0.85, lw=0.7))
        self.ax_m.text(1.008, 0.47, 'Esik: 0.50', fontsize=6, color=AMBER,
                       fontfamily='monospace', transform=self.ax_m.transAxes, ha='left')

        self.ax_m.text(0.005, 0.92, 'Saldiri Olasiligi (CRNN)', fontsize=7,
                       color=PURPLE, fontfamily='monospace', fontweight='bold',
                       transform=self.ax_m.transAxes, ha='left', va='top',
                       bbox=dict(boxstyle='round,pad=0.25', fc=BG, ec=PURPLE, alpha=0.75, lw=0.6))
        self.ax_m.set_ylim(-0.03, 1.06)
        self.ax_m.tick_params(colors=TXT, labelsize=6, length=2)
        self.ax_m.grid(True, color=GRID, alpha=0.5, lw=0.4)
        for sp in self.ax_m.spines.values():
            sp.set_color(GRID)

        # ---- Alt bilgi ----
        self.txt_ft = self.fig.text(0.5, 0.005, '', ha='center', fontsize=7.5,
                                     color=TXT, fontfamily='monospace')

        # ---- Lejant ----
        lg = [
            Line2D([0], [0], color=GREEN, lw=5, alpha=0.4, label='Normal: Temiz Veri'),
            Line2D([0], [0], color=RED, lw=5, alpha=0.4, label='Spoofing: Saldiri'),
            Line2D([0], [0], color=PURPLE, lw=2, label='Model: Saldiri Olasiligi'),
            Line2D([0], [0], color=AMBER, lw=1.5, ls='--', label='Karar Esigi: 0.50'),
        ]
        self.fig.legend(handles=lg, loc='upper center', ncol=4, fontsize=7.5,
                        facecolor=BG, edgecolor=GRID,
                        labelcolor=TXT_HI, bbox_to_anchor=(0.5, 0.993))

        # Gecici fill artistleri
        self._fills = []

        # Keyboard
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    # ------------------------------------------------------------------
    def _on_key(self, event):
        if event.key == ' ':
            self.paused = not self.paused

    # ------------------------------------------------------------------
    def _view(self):
        n = self.cursor
        ve = n
        vs = max(0, n - self.window)
        return vs, ve

    # ------------------------------------------------------------------
    def _update(self, frame):
        # Eski fill'leri temizle
        for f in self._fills:
            try: f.remove()
            except: pass
        self._fills.clear()

        # Yeni veri
        if not self.paused and self.cursor < self.N:
            oc = self.cursor
            nc = min(self.cursor + self.batch, self.N)
            cl = self.labels[oc:nc]
            cp = (self.probs[oc:nc] >= 0.5).astype(int)
            self.tp += int(np.sum((cp == 1) & (cl == 1)))
            self.fp += int(np.sum((cp == 1) & (cl == 0)))
            self.fn += int(np.sum((cp == 0) & (cl == 1)))
            self.tn += int(np.sum((cp == 0) & (cl == 0)))
            self.cursor = nc

        if self.cursor < 2:
            return

        vs, ve = self._view()
        nv = ve - vs
        if nv < 2:
            return

        x = np.arange(vs, ve)
        lv = self.labels[vs:ve]

        # Spoof mask
        spoof_mask = lv == 1
        norm_mask  = ~spoof_mask

        # NaN trick: spoofing satirlarini NaN yaparak cizgiyi kes
        nan_norm  = np.empty(nv)
        nan_spoof = np.empty(nv)

        # ---- Feature panelleri ----
        for p in self.panels:
            feat = p['feat']
            ax   = p['ax']
            vals = self.raw[feat][vs:ve]

            # Normal cizgi (spoofing'de NaN)
            nan_norm[:] = vals
            nan_norm[spoof_mask] = np.nan
            p['ln_norm'].set_data(x, nan_norm)

            # Spoofing cizgi (normal'de NaN)
            nan_spoof[:] = vals
            nan_spoof[norm_mask] = np.nan
            p['ln_spoof'].set_data(x, nan_spoof)

            ymn, ymx = vals.min(), vals.max()
            pad = max((ymx - ymn) * 0.08, 0.5)
            ax.set_ylim(ymn - pad, ymx + pad)
            ax.set_xlim(vs, max(ve, vs + 20))

            p['txt_v'].set_text(f'{vals[-1]:.2f}')

            # Spoofing arka plan — opakligi dusuk tek axvspan
            # Bolge bazli degil, hepsini tek imshow ile yapiyoruz
            # Ama imshow her karede degismeli — en hizlisi: koyu fill
            # Sadece buyuk bolgeler icin span
            regs = self._regions(lv, vs)
            for rs, re in regs:
                sp = ax.axvspan(rs, re, alpha=0.06, color=RED, zorder=0)
                self._fills.append(sp)

        # ---- Model paneli ----
        pv = self.probs[vs:ve]
        self.ln_p.set_data(x, pv)
        self.ln_pg.set_data(x, pv)
        self.ax_m.set_xlim(vs, max(ve, vs + 20))

        if len(pv) > 0:
            self.txt_pv.set_text(f'{pv[-1]:.3f}')

        if len(pv) > 1:
            fr = self.ax_m.fill_between(x, pv, 0.5, where=(pv >= 0.5),
                                         alpha=0.18, color=RED, interpolate=True)
            fg = self.ax_m.fill_between(x, pv, 0.5, where=(pv < 0.5),
                                         alpha=0.07, color=GREEN, interpolate=True)
            self._fills.extend([fr, fg])

        regs = self._regions(lv, vs)
        for rs, re in regs:
            sp = self.ax_m.axvspan(rs, re, alpha=0.05, color=RED, zorder=0)
            self._fills.append(sp)

        # Saldiri etiketleri
        det = self._regions((pv >= 0.5).astype(int), vs)
        for ds, de in det:
            if de - ds > 3:  # Cok kucuk bolgeler icin etiket koyma
                mid = (ds + de) / 2
                t = self.ax_m.text(mid, 0.93, 'SALDIRI', fontsize=5.5,
                                    fontweight='bold', color=RED, ha='center',
                                    fontfamily='monospace', alpha=0.55)
                self._fills.append(t)

        # ---- Alarm ----
        cp = self.probs[self.cursor - 1] if self.cursor > 0 else 0
        if cp >= 0.5:
            blink = '!! SPOOFING SALDIRISI TESPIT EDILDI !!'
            c = RED if frame % 3 != 0 else '#ff9999'
            self.txt_al.set_text(blink)
            self.txt_al.set_color(c)
            self.txt_al.set_bbox(dict(boxstyle='round,pad=0.5', fc=RED_BG, ec=c, alpha=0.95))
        else:
            self.txt_al.set_text('')
            self.txt_al.set_bbox(dict(boxstyle='round,pad=0.5', fc=RED_BG, ec=RED, alpha=0.0))

        # Duraklat gostergesi
        if self.paused:
            t = self.ax_h.text(0.5, 0.6, '|| DURAKLATILDI', fontsize=12,
                                fontweight='bold', color=AMBER,
                                transform=self.ax_h.transAxes, ha='center',
                                fontfamily='monospace')
            self._fills.append(t)

        # ---- Istatistik ----
        pct = self.cursor / self.N * 100
        sn = self.tp + self.fn
        dp = (self.tp / max(sn, 1)) * 100
        prec = self.tp / max(self.tp + self.fp, 1)
        rec  = self.tp / max(self.tp + self.fn, 1)
        f1   = 2*prec*rec / max(prec+rec, 1e-9)

        self.txt_st.set_text(
            f'Islenen: {self.cursor:,}/{self.N:,} ({pct:.0f}%)  |  '
            f'Normal: {self.tn+self.fp:,}  |  '
            f'Saldiri: {sn:,}  |  '
            f'Tespit: %{dp:.1f}')

        self.txt_ft.set_text(
            f'Dogru Tespit (TP): {self.tp}   '
            f'Yanlis Alarm (FP): {self.fp}   '
            f'Kacirilan (FN): {self.fn}   '
            f'Dogru Normal (TN): {self.tn}   |   '
            f'Hassasiyet: {prec:.3f}   '
            f'Yakalama: {rec:.3f}   '
            f'F1: {f1:.3f}')

    # ------------------------------------------------------------------
    def _regions(self, arr, offset):
        regs = []
        inside = False
        s = 0
        for i in range(len(arr)):
            if arr[i] >= 0.5 and not inside:
                s = i; inside = True
            elif arr[i] < 0.5 and inside:
                regs.append((s + offset, i + offset))
                inside = False
        if inside:
            regs.append((s + offset, len(arr) + offset))
        return regs

    def run(self):
        self.anim = FuncAnimation(
            self.fig, self._update,
            frames=(self.N // self.batch) + 500,
            interval=self.interval, repeat=False,
            blit=False, cache_frame_data=False)
        plt.show()


# ==============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--speed',    type=int, default=BATCH_SIZE)
    p.add_argument('--interval', type=int, default=INTERVAL_MS)
    p.add_argument('--window',   type=int, default=VIEW_WINDOW)
    args = p.parse_args()

    print("=" * 60)
    print("  IMEJE-BKZS Anti-Spoofing Canli Simulasyon v4")
    print("=" * 60)
    print()
    print("  Kontroller:")
    print("    SPACE                 Duraklat / Devam")
    print()

    model, scaler, df, features = load_all()
    print(f"  Veri: {len(df):,} satir  |  Hiz: {args.speed} satir/kare")

    sim = LiveSim(model, scaler, df, features,
                  batch=args.speed, interval=args.interval, window=args.window)
    print("\n  Baslatiliyor...\n")
    sim.run()


if __name__ == "__main__":
    main()
