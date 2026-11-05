# =============================================================================
# VERİ MADENCİLİĞİNE GİRİŞ – DÖNEM SONU PROJESİ
# Proje Başlığı : Boston Konut Fiyatı Çoklu Sınıflandırması (Ucuz, Orta, Pahalı)
# Algoritma     : Random Forest Classifier
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.metrics import confusion_matrix

# =============================================================================
# 1. VERİ YÜKLEME VE 3 SINIFLI HEDEF DEĞİŞKEN DÖNÜŞÜMÜ
# =============================================================================

df = pd.read_csv('data/Boston-house-price-data.csv')

# MEDV kolonunu istatistiksel olarak eşit büyüklükte 3 parçaya (tertiles) bölüyoruz
# 0: Ucuz, 1: Orta, 2: Pahalı
df['PRICE_CLASS'] = pd.qcut(df['MEDV'], q=3, labels=[0, 1, 2])

print("=" * 60)
print("VERİ SETİ GENEL BİLGİLERİ")
print("=" * 60)
print(f"Satır sayısı  : {df.shape[0]}")
print(f"Sütun sayısı  : {df.shape[1]}")
print(f"\nSınıf Dağılımı:\n{df['PRICE_CLASS'].value_counts().sort_index()}")

# =============================================================================
# 2. VERİ GÖRSELLEŞTİRME (Rapor Bölüm 3 için)
# =============================================================================

# --- Görsel 1: 3 Sınıfın Dağılımı Bar Plot ---
plt.figure(figsize=(6, 4))
sns.countplot(x='PRICE_CLASS', data=df, palette='viridis')
plt.title('Ev Fiyatı Sınıf Dağılımı (0: Ucuz, 1: Orta, 2: Pahalı)', fontsize=12)
plt.xlabel('Sınıf')
plt.ylabel('Frekans')
plt.tight_layout()
plt.savefig('viz1_class_distribution_3.png', dpi=100)
plt.show()

# --- Görsel 2: Korelasyon Isı Haritası ---
df_viz = df.drop(['MEDV', 'B', 'TAX'], axis=1) 
corr = df_viz.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Korelasyon Matrisi (Ön İşleme Öncesi)', fontsize=14)
plt.tight_layout()
plt.savefig('viz2_correlation.png', dpi=100)
plt.show()

# --- Görsel 3: Box Plot (LSTAT'a göre 3 Sınıfın Dağılımı) ---
plt.figure(figsize=(8, 5))
sns.boxplot(x='PRICE_CLASS', y='LSTAT', data=df, palette='Set2')
plt.title('Fiyat Sınıfına Göre Düşük Gelirli Nüfus Oranı (LSTAT)', fontsize=12)
plt.xlabel('Sınıf (0: Ucuz, 1: Orta, 2: Pahalı)')
plt.ylabel('LSTAT (%)')
plt.tight_layout()
plt.savefig('viz3_boxplot_3.png', dpi=100)
plt.show()

# =============================================================================
# 3. VERİ ÖN İŞLEME (Rapor Bölüm 4 için)
# =============================================================================

X = df.drop(['MEDV', 'PRICE_CLASS', 'B', 'TAX'], axis=1)
y = df['PRICE_CLASS']

# --- Standardizasyon ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Özellik Seçimi (SelectKBest – f_classif çoklu sınıfı otomatik destekler) ---
selector = SelectKBest(f_classif, k=7)
X_selected = selector.fit_transform(X_scaled, y)

selected_features = X.columns[selector.get_support()].tolist()

print("\n" + "=" * 60)
print("ÖZELLİK SEÇİMİ SONUÇLARI (Rapor Bölüm 4.2)")
print("=" * 60)
print(f"Başlangıç özellik sayısı : {X.shape[1]}")
print(f"Seçilen özellik sayısı   : {X_selected.shape[1]}")
print(f"Seçilen özellikler       : {selected_features}")

# =============================================================================
# 4. ALGORİTMA: RANDOM FOREST CLASSIFIER (Rapor Bölüm 5 için)
# =============================================================================

model = RandomForestClassifier(n_estimators=100, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Çoklu sınıflandırma (Multi-class) olduğu için metriklerde 'weighted' ortalama kullanıyoruz.
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision_weighted',
    'recall': 'recall_weighted',
    'f1': 'f1_weighted'
}

cv_results = cross_validate(model, X_selected, y, cv=kf, scoring=scoring)

print("\n" + "=" * 60)
print("5-FOLD CV ORTALAMA SONUÇLARI (Rapor Bölüm 5.3)")
print("=" * 60)
print(f"Doğruluk (Accuracy) : %{cv_results['test_accuracy'].mean() * 100:.2f}")
print(f"Kesinlik (Precision): %{cv_results['test_precision'].mean() * 100:.2f} (Weighted)")
print(f"Duyarlılık (Recall) : %{cv_results['test_recall'].mean() * 100:.2f} (Weighted)")
print(f"F1-Skoru            : %{cv_results['test_f1'].mean() * 100:.2f} (Weighted)")

print("\n" + "=" * 60)
print("5-FOLD CV DETAYLARI (Rapor Bölüm 5.4)")
print("=" * 60)
for i, acc in enumerate(cv_results['test_accuracy']):
    print(f"Fold {i+1}: %{acc * 100:.2f}")

# =============================================================================
# RAPOR BÖLÜM 5.2 İÇİN 3x3 KONFÜZYON MATRİSİ GÖRSELİ
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ucuz (0)', 'Orta (1)', 'Pahalı (2)'], 
            yticklabels=['Ucuz (0)', 'Orta (1)', 'Pahalı (2)'])
plt.title('3 Sınıflı Konfüzyon Matrisi', fontsize=14)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.tight_layout()
plt.savefig('viz4_confusion_matrix_3.png', dpi=100)
plt.show()

print("\n" + "=" * 60)
print("Proje başarıyla tamamlandı. Rapor için gerekli 3 Sınıflı görseller kaydedildi.")
print("=" * 60)