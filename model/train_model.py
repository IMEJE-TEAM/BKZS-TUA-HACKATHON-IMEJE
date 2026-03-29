import os
import sys


cuda_bin_path = os.path.join(os.getcwd(), '.venv', 'Lib', 'site-packages', 'nvidia', 'cudnn', 'bin')
if os.path.exists(cuda_bin_path):
    os.environ['PATH'] = cuda_bin_path + os.pathsep + os.environ['PATH']

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

print("-" * 50)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU bulundu: {len(gpus)} adet")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU CUDA ile egitim yapilacak.")
    except RuntimeError as e:
        print(f"GPU Yapilandirma Hatasi: {e}")
else:
    print("GPU bulunamadi, CPU ile devam ediliyor.")
print("-" * 50)

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense,
    Dropout, BatchNormalization, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("Anti-Spoofing Modeli Baslatiliyor...")
print("-" * 50)


file_normal = "egitim_normal.csv"
file_spoof = "egitim_spoofing.csv"
file_test = "test_senaryosu.csv"

for f in [file_normal, file_spoof, file_test]:
    if not os.path.exists(f):
        print(f"HATA: '{f}' bulunamadi! Once extract_data.py calistirin.")
        sys.exit(1)

print("Veri setleri okunuyor...")
df_normal = pd.read_csv(file_normal)
df_spoof = pd.read_csv(file_spoof)
df_test = pd.read_csv(file_test)


features = [c for c in df_normal.columns if c != 'Label']

print(f"Feature'lar: {features}")
print(f"Normal egitim: {len(df_normal)} satir")
print(f"Spoofing egitim: {len(df_spoof)} satir")
print(f"Test senaryosu: {len(df_test)} satir")


print("\nNormalizasyon uygulaniyor...")
scaler = StandardScaler()
all_train_features = pd.concat(
    [df_normal[features], df_spoof[features]], ignore_index=True
)
scaler.fit(all_train_features)

with open("crnn_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

scaled_normal = scaler.transform(df_normal[features])
scaled_spoof = scaler.transform(df_spoof[features])
scaled_test = scaler.transform(df_test[features])

print(f"Scaler ortalamalar: {scaler.mean_}")
print(f"Scaler std: {scaler.scale_}")


TIME_STEPS = 30

def create_sequences(data, labels, time_steps):
    """Sliding window ile ardisik bloklar olusturur."""
    X, y = [], []
    labels_arr = np.array(labels)
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(labels_arr[i + time_steps])
    return np.array(X), np.array(y)

print(f"\n{TIME_STEPS} adimlik pencereler olusturuluyor...")

X_norm, y_norm = create_sequences(scaled_normal, df_normal['Label'], TIME_STEPS)
X_spoof, y_spoof = create_sequences(scaled_spoof, df_spoof['Label'], TIME_STEPS)
X_test_seq, y_test_seq = create_sequences(scaled_test, df_test['Label'], TIME_STEPS)

X_train_all = np.vstack((X_norm, X_spoof))
y_train_all = np.hstack((y_norm, y_spoof))
X_train, y_train = shuffle(X_train_all, y_train_all, random_state=42)

n_features = len(features)
print(f"Egitim pencere sayisi: {len(X_train)}")
print(f"Test pencere sayisi: {len(X_test_seq)}")
print(f"Giris boyutu: ({TIME_STEPS}, {n_features})")


print("\nModel mimarisi olusturuluyor...")

model = Sequential([
    Input(shape=(TIME_STEPS, n_features)),

    # 1D CNN - yerel oruntu cikarimi
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    BatchNormalization(),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # Bidirectional LSTM - zamansal bagimliliklar
    Bidirectional(LSTM(128, return_sequences=False)),
    Dropout(0.4),

    # Siniflandirici
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


print("\nEgitim basliyor...")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=1024,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Egitim Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validasyon Loss', linewidth=2, linestyle='--')
axes[0].set_title('Model - Loss Egrisi')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['accuracy'], label='Egitim Accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validasyon Accuracy', linewidth=2, linestyle='--')
axes[1].set_title('Model - Accuracy Egrisi')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('egitim_grafikleri.png', dpi=150)
plt.close()
print("Egitim grafikleri 'egitim_grafikleri.png' olarak kaydedildi.")

print("\n" + "=" * 50)
print("TEST SENARYOSU DEGERLENDIRMESI")
print("=" * 50)

loss, accuracy = model.evaluate(X_test_seq, y_test_seq, verbose=0)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: %{accuracy * 100:.2f}")

y_pred_prob = model.predict(X_test_seq, verbose=0).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)
y_true = y_test_seq.astype(int)

print("\nSiniflandirma Raporu:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Spoofing']))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(f"  TN={cm[0][0]:>7d}  FP={cm[0][1]:>7d}")
print(f"  FN={cm[1][0]:>7d}  TP={cm[1][1]:>7d}")

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Spoofing'],
            yticklabels=['Normal', 'Spoofing'], annot_kws={"size": 16}, ax=ax)
ax.set_xlabel('Tahmin Edilen', fontsize=12)
ax.set_ylabel('Gercek Deger', fontsize=12)
ax.set_title(f'Confusion Matrix (Accuracy: %{accuracy*100:.2f})', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("confusion_matrix.png kaydedildi.")

# ==============================================================================
# 9. GRAFIK 2: ROC CURVE + AUC
# ==============================================================================
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='#2196F3', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random (AUC = 0.50)')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve', fontsize=14)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.close()
print(f"roc_curve.png kaydedildi. (AUC: {roc_auc:.4f})")

# ==============================================================================
# 10. GRAFIK 3: PRECISION-RECALL CURVE
# ==============================================================================
precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_prob)
ap_score = average_precision_score(y_true, y_pred_prob)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(recall_vals, precision_vals, color='#FF5722', lw=2, label=f'PR Curve (AP = {ap_score:.4f})')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve', fontsize=14)
ax.legend(loc='lower left', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=150)
plt.close()
print(f"precision_recall_curve.png kaydedildi. (AP: {ap_score:.4f})")


fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(y_pred_prob[y_true == 0], bins=50, alpha=0.6, color='#4CAF50', label='Normal', density=True)
ax.hist(y_pred_prob[y_true == 1], bins=50, alpha=0.6, color='#F44336', label='Spoofing', density=True)
ax.axvline(x=0.5, color='black', linestyle='--', lw=1.5, label='Karar Siniri (0.5)')
ax.set_xlabel('Tahmin Olasiligi (Spoofing)', fontsize=12)
ax.set_ylabel('Yogunluk', fontsize=12)
ax.set_title('Tahmin Olasilik Dagilimi', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tahmin_dagilimi.png', dpi=150)
plt.close()
print("tahmin_dagilimi.png kaydedildi.")


fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for idx, feat in enumerate(features):
    if idx >= 8:
        break
    ax = axes[idx]
    normal_vals = df_normal[feat].values
    spoof_vals = df_spoof[feat].values
    ax.hist(normal_vals, bins=60, alpha=0.5, color='#4CAF50', label='Normal', density=True)
    ax.hist(spoof_vals, bins=60, alpha=0.5, color='#F44336', label='Spoofing', density=True)
    ax.set_title(feat, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

plt.suptitle('Feature Dagilimi: Normal vs Spoofing', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('feature_dagilimi.png', dpi=150, bbox_inches='tight')
plt.close()
print("feature_dagilimi.png kaydedildi.")


fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

display_range = min(3000, len(y_true))
x_axis = range(display_range)

# Gercek etiketler
axes[0].fill_between(x_axis, y_true[:display_range], alpha=0.4, color='#2196F3')
axes[0].set_title('Gercek Etiketler (Normal=0, Spoofing=1)', fontsize=12)
axes[0].set_ylabel('Etiket')
axes[0].set_ylim(-0.1, 1.1)
axes[0].grid(True, alpha=0.3)


axes[1].fill_between(x_axis, y_pred[:display_range], alpha=0.4, color='#F44336')
axes[1].set_title('Model Tahminleri', fontsize=12)
axes[1].set_ylabel('Etiket')
axes[1].set_ylim(-0.1, 1.1)
axes[1].grid(True, alpha=0.3)


axes[2].plot(x_axis, y_pred_prob[:display_range], color='#9C27B0', alpha=0.7, lw=0.8)
axes[2].axhline(y=0.5, color='black', linestyle='--', lw=1, alpha=0.5)
axes[2].fill_between(x_axis, y_pred_prob[:display_range], 0.5,
                     where=(y_pred_prob[:display_range] >= 0.5),
                     alpha=0.2, color='red', label='Spoofing Bolgesi')
axes[2].fill_between(x_axis, y_pred_prob[:display_range], 0.5,
                     where=(y_pred_prob[:display_range] < 0.5),
                     alpha=0.2, color='green', label='Normal Bolgesi')
axes[2].set_title('Tahmin Olasiligi (Surekli)', fontsize=12)
axes[2].set_xlabel('Zaman Adimi')
axes[2].set_ylabel('P(Spoofing)')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bolgesel_analiz.png', dpi=150)
plt.close()
print("bolgesel_analiz.png kaydedildi.")


fig, ax = plt.subplots(figsize=(18, 4))

errors = (y_pred != y_true).astype(int)
fp_mask = (y_pred == 1) & (y_true == 0)  
fn_mask = (y_pred == 0) & (y_true == 1)  

display_range_err = min(5000, len(y_true))
x_err = range(display_range_err)

ax.fill_between(x_err, y_true[:display_range_err], alpha=0.15, color='blue', label='Gercek Spoofing')
ax.scatter(np.where(fp_mask[:display_range_err])[0], np.ones(fp_mask[:display_range_err].sum()) * 0.8,
           c='red', s=8, alpha=0.7, label=f'False Positive ({fp_mask.sum()})')
ax.scatter(np.where(fn_mask[:display_range_err])[0], np.ones(fn_mask[:display_range_err].sum()) * 0.2,
           c='orange', s=8, alpha=0.7, label=f'False Negative ({fn_mask.sum()})')
ax.set_title('Hata Analizi: Yanlis Tahminlerin Konumu', fontsize=13)
ax.set_xlabel('Zaman Adimi')
ax.set_ylabel('Etiket')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 1.1)
plt.tight_layout()
plt.savefig('hata_analizi.png', dpi=150)
plt.close()
print("hata_analizi.png kaydedildi.")


model.save("crnn_model.h5")
print(f"\nModel 'crnn_model.h5' olarak kaydedildi.")
print(f"Scaler 'crnn_scaler.pkl' olarak kaydedildi.")

print("\n" + "=" * 50)
print("TAMAMLANDI! Olusturulan dosyalar:")
print("  - crnn_model.h5          (Model)")
print("  - crnn_scaler.pkl        (Scaler)")
print("  - egitim_grafikleri.png   (Loss + Accuracy)")
print("  - confusion_matrix.png   (Confusion Matrix)")
print("  - roc_curve.png          (ROC + AUC)")
print("  - precision_recall_curve.png (Precision-Recall)")
print("  - tahmin_dagilimi.png    (Olasilik Histogrami)")
print("  - feature_dagilimi.png   (Feature Karsilastirma)")
print("  - bolgesel_analiz.png    (Zaman Serisi Analizi)")
print("  - hata_analizi.png       (FP/FN Konumlari)")
print("=" * 50)
