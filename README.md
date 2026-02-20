<div align="center">

<!-- Animated Wave Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&height=210&color=0:16a34a,100:22c55e&text=Intro%20to%20Machine%20Learning&fontSize=50&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Kaggle%20Learn%20Notes%20%7C%20Iowa%20Housing%20%7C%20scikit-learn%20Quick%20Reference&descAlignY=58&descSize=18" />

<!-- Typing Animation -->
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&duration=2600&pause=700&color=16A34A&center=true&vCenter=true&width=940&lines=EDA+with+pandas+%E2%9C%85+Decision+Tree+%E2%9C%85+Random+Forest+%E2%9C%85;Train%2FValidation+Split+%E2%9C%85+MAE+%E2%9C%85+Tuning+max_leaf_nodes+%E2%9C%85;From+first+model+to+Kaggle+submission+%F0%9F%8F%86;Cheat+Sheet+%2B+Troubleshooting+%2B+Resources" />

<br/>

<!-- Hero Image -->
<img src="https://images.pexels.com/photos/590016/pexels-photo-590016.jpeg?auto=compress&cs=tinysrgb&w=1200&h=280&fit=crop" alt="Machine Learning" width="100%" style="border-radius:14px;" />

<br/><br/>

<b>📘 Intro to Machine Learning — Panduan Lengkap & Referensi Cepat</b><br/>
<i>Catatan ringkas + contoh kode untuk menyelesaikan course Kaggle Learn “Intro to Machine Learning”.</i><br/><br/>

<!-- Badges -->
<img src="https://img.shields.io/badge/Python-16a34a?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/pandas-0f172a?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-22c55e?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/Kaggle-2563eb?style=for-the-badge&logo=kaggle&logoColor=white"/>

<br/><br/>

<p>
<b>Target:</b> cepat paham alur ML dasar → bikin model pertama → validasi MAE → Random Forest → submit ke Kaggle.
</p>

</div>

---

## 🧭 Table of Contents
- [📌 Ringkasan](#-ringkasan)
- [🚀 Quick Start](#-quick-start)
- [🧠 Workflow ML Singkat](#-workflow-ml-singkat)
- [📚 Materi Lengkap](#-materi-lengkap)
- [✅ License](#-license)

---

## 📌 Ringkasan

Yang kamu dapat dari README ini:

- ✅ Ringkasan setiap lesson (1–7)
- ✅ Contoh kode siap copy-paste (pandas + scikit-learn)
- ✅ Cheat sheet, troubleshooting, dan resources
- ✅ Referensi untuk dataset **Iowa Housing (Ames)**

---

## 🚀 Quick Start

### Opsi 1 — Kaggle Notebooks (paling gampang)
1. Buka course Kaggle Learn “Intro to Machine Learning”
2. Jalankan notebook lesson → copy bagian kode dari sini saat perlu

### Opsi 2 — Local Jupyter
```bash
pip install pandas scikit-learn numpy matplotlib seaborn jupyter
jupyter notebook
```

---

## 🧠 Workflow ML Singkat

```mermaid
flowchart LR
  A[Load data] --> B[Explore with pandas]
  B --> C[Choose target (y) + features (X)]
  C --> D[Train model]
  D --> E[Validate with MAE]
  E --> F[Tune / switch model]
  F --> G[Train full data]
  G --> H[Predict test + submit Kaggle]
```

---

## 📚 Materi Lengkap

<details open>
<summary><b>📄 Klik untuk lihat catatan lengkap (Full Notes)</b></summary>

<br/>

# Intro to Machine Learning — Panduan Lengkap & Referensi Cepat

**Version:** 1.0  
**Last Updated:** 2025-12-12  
**Language:** Bahasa Indonesia & English  
**Course:**  Kaggle Learn — Intro to Machine Learning  
**Dataset:**  Iowa Housing Prices (Ames Housing Dataset)

---

## 📚 Daftar Isi

1. [Ringkasan Course](#ringkasan-course)
2. [Prerequisite](#prerequisite)
3. [Setup Environment](#setup-environment)
4. [Lesson 1: How Models Work](#lesson-1-how-models-work)
5. [Lesson 2: Basic Data Exploration](#lesson-2-basic-data-exploration)
6. [Lesson 3: Your First Machine Learning Model](#lesson-3-your-first-machine-learning-model)
7. [Lesson 4: Model Validation](#lesson-4-model-validation)
8. [Lesson 5: Underfitting and Overfitting](#lesson-5-underfitting-and-overfitting)
9. [Lesson 6: Random Forests](#lesson-6-random-forests)
10. [Lesson 7: Machine Learning Competitions](#lesson-7-machine-learning-competitions)
11. [Cheat Sheet](#cheat-sheet)
12. [Troubleshooting](#troubleshooting)
13. [Next Steps](#next-steps)
14. [Resources](#resources)

---

## 🎯 Ringkasan Course

**Intro to Machine Learning** adalah course pemula untuk memahami konsep dasar machine learning dan membangun model prediktif menggunakan **scikit-learn**. 

### Apa yang Dipelajari: 

✅ Konsep dasar machine learning (supervised learning, regression)  
✅ Eksplorasi data dengan **pandas**  
✅ Membangun model **Decision Tree** & **Random Forest**  
✅ Evaluasi model dengan **Mean Absolute Error (MAE)**  
✅ Train/validation split untuk menghindari overfitting  
✅ Hyperparameter tuning untuk improve akurasi  
✅ Submit prediksi ke **Kaggle competition**

### Tools & Libraries:

- **Python 3.x**
- **pandas** — data manipulation
- **scikit-learn** — machine learning models
- **numpy** — numerical operations
- **Kaggle Notebooks** — cloud environment (opsional)

### Dataset:

**Iowa Housing Prices** — prediksi harga rumah berdasarkan 79 features (luas tanah, tahun dibangun, jumlah kamar, dll).

---

## 📋 Prerequisite

### Knowledge: 

- **Python basics** (variables, functions, loops, conditionals)
- **Basic math** (mean, median, algebra dasar)
- **Curiosity & persistence** 🚀

### Tidak Perlu: 

❌ Background matematika/statistik advanced  
❌ Pengalaman coding sebelumnya (tapi helpful)  
❌ Pengetahuan machine learning (ini course pemula)

### Rekomendasi Setup:

**Opsi 1: Kaggle Notebooks (Easiest)**
- Gratis, cloud-based, dataset sudah tersedia
- Link: https://www.kaggle.com/learn/intro-to-machine-learning

**Opsi 2: Local Jupyter Notebook**
```bash
# Install dependencies
pip install pandas scikit-learn numpy jupyter

# Download dataset dari Kaggle
# Jalankan Jupyter
jupyter notebook
```

---

## 🛠 Setup Environment

### Install Libraries (Local):

```bash
pip install pandas scikit-learn numpy matplotlib seaborn jupyter
```

### Import Libraries (Standard):

```python
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
```

### Load Dataset:

```python
# Path ke file dataset
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

# Load data dengan pandas
home_data = pd.read_csv(iowa_file_path)

# Preview data
print(home_data. head())
print(home_data.describe())
```

---

## 📖 Lesson 1:  How Models Work

### Konsep Dasar

**Machine Learning** = membuat komputer "belajar" dari data untuk membuat prediksi/keputusan.

### Supervised Learning: 

- **Input:** Data dengan label/target (contoh: harga rumah)
- **Output:** Model yang bisa prediksi label untuk data baru
- **Contoh:** Prediksi harga rumah berdasarkan features (luas, lokasi, tahun dibangun)

### Decision Tree:

Model yang membuat keputusan dengan "memotong" data berdasarkan features.

**Contoh Decision Tree (simplified):**

```
                    [All Houses]
                    /          \
        [YearBuilt > 2000? ]     [YearBuilt ≤ 2000?]
         /         \             /           \
   [Large Area] [Small]    [Good Cond]  [Poor Cond]
   → $300k      → $200k    → $180k      → $120k
```

### Key Terms:

| Term | Definisi |
|------|----------|
| **Features (X)** | Input variables (mis.  luas tanah, tahun dibangun) |
| **Target (y)** | Variable yang ingin diprediksi (mis. harga rumah) |
| **Training Data** | Data untuk melatih model |
| **Prediction** | Output model untuk data baru |
| **Model** | Algoritma yang "belajar" pola dari data |

---

## 📊 Lesson 2:  Basic Data Exploration

### Load & Inspect Data

```python
import pandas as pd

# Load data
home_data = pd.read_csv('train.csv')

# Lihat 5 baris pertama
print(home_data.head())

# Lihat statistik deskriptif
print(home_data.describe())

# Lihat semua kolom
print(home_data.columns)

# Lihat info dataset (tipe data, missing values)
print(home_data.info())
```

### Key Pandas Methods:

| Method | Fungsi |
|--------|--------|
| `.head(n)` | Lihat n baris pertama (default 5) |
| `.tail(n)` | Lihat n baris terakhir |
| `.describe()` | Statistik deskriptif (mean, std, min, max, dll) |
| `.info()` | Info tipe data & missing values |
| `.shape` | Dimensi dataset (rows, columns) |
| `.columns` | Daftar nama kolom |
| `.isnull().sum()` | Hitung missing values per kolom |

### Contoh Output `.describe()`:

```
       SalePrice    LotArea   YearBuilt
count  1460.00      1460.00   1460.00
mean   180921.20    10516.83  1971.27
std    79442.50     9981.26   30.20
min    34900.00     1300.00   1872.00
25%    129975.00    7553.50   1954.00
50%    163000.00    9478.50   1973.00
75%    214000.00    11601.50  2000.00
max    755000.00    215245.00 2010. 00
```

---

## 🤖 Lesson 3:  Your First Machine Learning Model

### Workflow: 

1. **Define Target (y)** — kolom yang ingin diprediksi
2. **Choose Features (X)** — kolom input untuk model
3. **Define Model** — pilih algoritma (mis.   Decision Tree)
4. **Fit Model** — latih model dengan data
5. **Predict** — buat prediksi

### Kode Lengkap:

```python
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# 1. Load data
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)

# 2. Define Target (y)
y = home_data.SalePrice

# 3. Choose Features (X)
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 
            'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# 4. Define Model
iowa_model = DecisionTreeRegressor(random_state=1)

# 5. Fit Model
iowa_model. fit(X, y)

# 6. Predict (pada data training untuk demo)
predictions = iowa_model.predict(X)
print("Predictions:", predictions[: 5])
print("Actual values:", y. head().values)
```

### Penjelasan: 

- **`random_state=1`** → memastikan hasil reproducible (sama setiap kali dijalankan)
- **`.fit(X, y)`** → melatih model dengan features (X) dan target (y)
- **`.predict(X)`** → membuat prediksi

### ⚠️ Catatan Penting: 

Prediksi pada **training data** akan terlihat sangat akurat (bahkan sempurna) → ini **misleading**!  
Untuk evaluasi yang benar, gunakan **validation data** (Lesson 4).

---

## ✅ Lesson 4:  Model Validation

### Masalah:   In-Sample Score

**In-sample evaluation** = evaluasi model pada data yang sama yang dipakai untuk training → **menyesatkan** (model bisa overfitting).

### Solusi:  Train-Test Split

Pisahkan data menjadi: 
- **Training set** (75%) → untuk melatih model
- **Validation set** (25%) → untuk evaluasi

### Kode: 

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Split data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define & fit model (hanya dengan training data)
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)

# Predict pada validation data
val_predictions = iowa_model.predict(val_X)

# Calculate MAE (Mean Absolute Error)
val_mae = mean_absolute_error(val_y, val_predictions)
print("Validation MAE:", val_mae)
```

### Mean Absolute Error (MAE):

**Formula:**
```
MAE = (1/n) × Σ |actual - predicted|
```

**Interpretasi:**
> "Secara rata-rata, prediksi model meleset sekitar $X dari harga aktual."

**Contoh:**
- MAE = 25,000 → rata-rata error $25,000
- Semakin rendah MAE, semakin baik model

### Perbandingan In-Sample vs Out-of-Sample:

| Metric | Training Data | Validation Data |
|--------|---------------|-----------------|
| **MAE** | ~500 (sangat rendah) | ~29,000 (realistis) |
| **Interpretasi** | Model "mengingat" training data | Model performance pada data baru |
| **Kesimpulan** | Misleading (overfitting) | ✅ Realistis |

---

## 🎯 Lesson 5:  Underfitting and Overfitting

### Konsep: 

**Underfitting** = model terlalu sederhana → gagal menangkap pola penting → MAE tinggi  
**Overfitting** = model terlalu kompleks → "mengingat" training data → MAE rendah di training, tinggi di validation

### Grafik: 

```
MAE
 |
 |   Underfitting
 |       \
 |        \_____ Sweet Spot (Optimal)
 |              \
 |               \_____ Overfitting
 |_________________________ Model Complexity
    (shallow tree)          (deep tree)
```

### Hyperparameter:   `max_leaf_nodes`

**`max_leaf_nodes`** = jumlah maksimal leaves (kelompok akhir) di Decision Tree.

- **Rendah** (mis. 5) → shallow tree → underfitting
- **Tinggi** (mis. 5000) → deep tree → overfitting
- **Optimal** (mis.  100) → sweet spot → MAE terendah

### Experiment:  Find Optimal `max_leaf_nodes`

```python
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

# Test berbagai nilai
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) 
          for leaf_size in candidate_max_leaf_nodes}

# Cari yang optimal
best_tree_size = min(scores, key=scores.get)
print(f"Optimal max_leaf_nodes: {best_tree_size}")
print(f"Best MAE: {scores[best_tree_size]: ,.0f}")

# Fit final model dengan parameter optimal
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
final_model.fit(X, y)  # Fit dengan SEMUA data
```

### Hasil Contoh:

```
max_leaf_nodes=5    → MAE:  35,044 (underfitting)
max_leaf_nodes=25   → MAE:  29,016
max_leaf_nodes=50   → MAE: 27,405
max_leaf_nodes=100  → MAE: 27,282 ✅ OPTIMAL
max_leaf_nodes=250  → MAE: 27,893
max_leaf_nodes=500  → MAE: 29,454 (overfitting)
```

---

## 🌲 Lesson 6:  Random Forests

### Masalah dengan Single Decision Tree: 

Bahkan setelah tuning, Decision Tree punya keterbatasan:
- Sensitif terhadap small changes di data
- Trade-off sulit antara underfitting & overfitting

### Solusi:  Random Forest

**Random Forest** = ensemble dari **banyak Decision Trees** (default:  100 trees).

### Cara Kerja:

1. Build 100 trees dengan: 
   - Random subset data (bootstrap sampling)
   - Random subset features
2. Setiap tree membuat prediksi
3. **Final prediction = rata-rata** prediksi semua trees

### Keunggulan: 

✅ **Lebih akurat** dari single Decision Tree  
✅ **Mengatasi overfitting** (averaging mengurangi variance)  
✅ **Robust dengan default parameters** (tidak perlu tuning ekstensif)  
✅ **"Just works"** — good performance out-of-the-box

### Kode:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Define Random Forest model
forest_model = RandomForestRegressor(random_state=1)

# Fit model
forest_model.fit(train_X, train_y)

# Predict
forest_preds = forest_model.predict(val_X)

# Evaluate
forest_mae = mean_absolute_error(val_y, forest_preds)
print("Random Forest MAE:", forest_mae)
```

### Perbandingan:   Decision Tree vs Random Forest

| Model | Validation MAE | Notes |
|-------|----------------|-------|
| **Decision Tree** (tuned) | ~27,282 | Setelah tuning `max_leaf_nodes` |
| **Random Forest** (default) | ~21,857 | **~20% lebih baik**, tanpa tuning!  |

### Tuning Random Forest (Opsional):

```python
# Custom parameters
forest_model_tuned = RandomForestRegressor(
    n_estimators=200,      # 200 trees (default:  100)
    max_depth=15,          # max depth per tree
    min_samples_split=5,
    random_state=1
)
forest_model_tuned.fit(train_X, train_y)
```

**Parameter Penting:**
- `n_estimators` — jumlah trees (lebih banyak = lebih akurat, tapi lebih lambat)
- `max_depth` — kedalaman maksimal tiap tree (kontrol overfitting)
- `min_samples_split` — minimum samples untuk split node

---

## 🏆 Lesson 7:  Machine Learning Competitions

### Submit ke Kaggle Competition

#### Step 1:  Train Model dengan SEMUA Data

```python
# Fit model dengan semua training data (bukan hanya train_X)
rf_model_on_full_data = RandomForestRegressor(random_state=1)
rf_model_on_full_data.fit(X, y)  # ← SEMUA data
```

**Catatan:** Validation data sudah selesai tugasnya (untuk tuning). Final model harus belajar dari **semua data** untuk maksimalkan performa.

#### Step 2:  Load Test Data & Predict

```python
# Load test data
test_data = pd.read_csv('../input/test.csv')

# Pilih features yang sama
test_X = test_data[features]

# Predict
test_preds = rf_model_on_full_data.predict(test_X)
```

#### Step 3:  Save Predictions

```python
# Format submission
output = pd.DataFrame({
    'Id': test_data. Id,
    'SalePrice': test_preds
})

# Save to CSV
output.to_csv('submission.csv', index=False)
print("Submission file created!")
```

#### Step 4:  Submit di Kaggle

1. Join competition:  [Housing Prices Competition](https://www.kaggle.com/c/home-data-for-ml-course)
2. Save & Run notebook ("Save Version" → "Save and Run All")
3. Open in Viewer
4. Tab "Data" → klik "submission.csv" → "Submit"

### Improve Model (Naik Leaderboard):

#### 1. Tambahkan Features

**Baseline (7 features):**
```python
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 
            'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
```

**Improved (25 features):**
```python
features = [
    'MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
    'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
    'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
    'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
    'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'
]
```

**Top Features untuk Akurasi:**
- `OverallQual` ⭐⭐⭐ — kualitas keseluruhan (1-10)
- `GrLivArea` ⭐⭐⭐ — luas living area (sq ft)
- `YearBuilt` ⭐⭐ — tahun dibangun
- `TotRmsAbvGrd` ⭐ — total ruangan

#### 2. Tuning Hyperparameters

```python
rf_model_on_full_data = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=1
)
```

#### 3. Try XGBoost (Advanced)

```python
from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=1)
xgb_model.fit(X, y)
test_preds = xgb_model.predict(test_X)
```

### Expected Results:

| Model | Validation MAE | Leaderboard Score |
|-------|----------------|-------------------|
| Baseline (7 features) | ~22,000 | ~26,000 |
| Improved (25 features) | ~17,000 | ~20,000 |
| Tuned + 25 features | ~15,000 | ~18,000 |

---

## 📝 Cheat Sheet

### Pandas Essentials

```python
import pandas as pd

# Load data
df = pd. read_csv('file.csv')

# Inspect
df.head()                  # 5 baris pertama
df.describe()              # Statistik deskriptif
df. info()                  # Info tipe data & missing values
df.columns                 # Nama kolom
df.shape                   # (rows, columns)

# Select
df['column']               # Select 1 kolom (Series)
df[['col1', 'col2']]       # Select multiple kolom (DataFrame)
```

### Scikit-Learn Workflow

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. Split data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# 2. Define model
model = RandomForestRegressor(random_state=1)

# 3. Fit model
model. fit(train_X, train_y)

# 4. Predict
predictions = model.predict(val_X)

# 5. Evaluate
mae = mean_absolute_error(val_y, predictions)
print("MAE:", mae)
```

### Decision Tree vs Random Forest

| Aspect | Decision Tree | Random Forest |
|--------|---------------|---------------|
| **Speed** | ⚡ Cepat (single tree) | 🐢 Lebih lambat (100 trees) |
| **Accuracy** | 📊 Sedang | 🎯 Tinggi |
| **Overfitting** | ⚠️ Mudah overfit | ✅ Robust |
| **Interpretability** | 📖 Mudah dijelaskan | 🔒 Black-box |
| **Tuning** | 🔧 Perlu tuning ekstensif | ✅ Default sudah bagus |

---

## 🚨 Troubleshooting

### Error:   `KeyError: 'column_name'`

**Penyebab:** Nama kolom salah atau tidak ada di dataset.  
**Solusi:**
```python
# Cek nama kolom yang benar
print(home_data.columns)

# Pastikan ejaan & huruf besar/kecil sesuai
y = home_data. SalePrice  # ✅ Benar
y = home_data. saleprice  # ❌ Salah (case-sensitive)
```

### Error:  `NameError: name 'X' is not defined`

**Penyebab:** Belum menjalankan cell yang define X dan y.  
**Solusi:** Jalankan cell setup secara berurutan dari atas.

### Error:  `ValueError: Input contains NaN`

**Penyebab:** Ada missing values (NaN) di data.  
**Solusi:**
```python
# Opsi 1: Hapus baris dengan missing values
home_data = home_data.dropna(axis=0)

# Opsi 2: Gunakan features tanpa missing values
# (lihat list 25 features yang aman di Lesson 7)
```

### MAE Sangat Tinggi (> 50,000)

**Penyebab:** Model underfitting atau features kurang informatif.  
**Solusi:**
- Tambahkan lebih banyak features (gunakan 25 features dari Lesson 7)
- Tuning hyperparameters (`max_depth`, `n_estimators`)
- Coba Random Forest (biasanya lebih baik dari Decision Tree)

### Validation MAE ≪ Leaderboard Score

**Penyebab:** Overfitting — model terlalu spesifik pada training data.  
**Solusi:**
- Kurangi kompleksitas model (`max_depth`, `n_estimators`)
- Gunakan cross-validation (advanced topic, pelajari di Intermediate ML)

---

## 🚀 Next Steps

### 1. Download Sertifikat

✅ Selesaikan semua 7 exercises  
✅ Klik "Get Certificate" di halaman course  
✅ Upload ke LinkedIn (Licenses & Certifications)

### 2. Lanjut Course Berikutnya

#### **Intermediate Machine Learning** (Recommended)
- Handle missing values
- Categorical variables (one-hot encoding)
- Pipelines
- **XGBoost** (model lebih powerful)
- Cross-validation

#### **Pandas**
- Deep dive data manipulation
- Merging, grouping, pivoting
- Time series
- Data cleaning

#### **Feature Engineering**
- Create new features dari existing data
- Feature selection
- Dimensionality reduction (PCA)

#### **Data Visualization**
- Matplotlib, Seaborn
- Exploratory Data Analysis (EDA)
- Storytelling dengan data

### 3. Build Portfolio Project

**Ide Project:**
- Predictive maintenance (IoT sensor data → predict failure)
- Stock price prediction
- Customer churn prediction
- House price prediction (improve Kaggle submission)

**Template Project:**
1. Load & explore data (EDA)
2. Feature engineering
3. Train multiple models (Decision Tree, Random Forest, XGBoost)
4. Compare MAE
5. Hyperparameter tuning
6. Final model + visualization
7. Publish di GitHub + README lengkap

### 4. Join Kaggle Competitions

**Beginner-Friendly Competitions:**
- Titanic (classification)
- House Prices (regression) ← Anda sudah mulai ini!
- Digit Recognizer (computer vision)

**Benefits:**
- Practice dengan real datasets
- Learn dari kernels/notebooks orang lain
- Build reputation (medals di profil)

---

## 📚 Resources

### Official Documentation

- **Scikit-Learn:** https://scikit-learn.org/stable/
- **Pandas:** https://pandas.pydata.org/docs/
- **Kaggle Learn:** https://www.kaggle.com/learn

### Books (Recommended)

1. **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** — Aurélien Géron  
   → Best practical ML book (Python)

2. **"Python for Data Analysis"** — Wes McKinney (creator of pandas)  
   → Deep dive pandas

3. **"The Elements of Statistical Learning"** — Hastie, Tibshirani, Friedman  
   → Theory (advanced, for S2/S3)

### Online Courses

- **Andrew Ng — Machine Learning (Coursera)** — Classic, foundational
- **Fast.ai — Practical Deep Learning** — Top-down approach
- **DataCamp / Coursera — Python for Data Science**

### Communities

- **Kaggle Forums & Discussions**
- **r/MachineLearning** (Reddit)
- **Stack Overflow** — untuk troubleshooting

---

## 📊 Summary Metrics

### Course Completion Stats: 

| Metric | Value |
|--------|-------|
| **Lessons** | 7 |
| **Exercises** | 7 |
| **Estimated Time** | 3-5 hours |
| **Difficulty** | Beginner |
| **Certificate** | ✅ Yes (free) |

### Model Performance (Iowa Housing):

| Model | Validation MAE | Notes |
|-------|----------------|-------|
| Baseline Decision Tree | ~29,000 | No tuning |
| Tuned Decision Tree | ~27,282 | `max_leaf_nodes=100` |
| Random Forest (default) | ~21,857 | **Best default** |
| Random Forest (25 features) | ~17,000 | Improved features |

---

## ✅ Checklist Completion

Gunakan checklist ini untuk track progress Anda:

- [ ] Lesson 1: Pahami konsep Decision Tree
- [ ] Lesson 2: Load & explore data dengan pandas
- [ ] Exercise 2: Hitung average lot size & newest home age
- [ ] Lesson 3: Build first Decision Tree model
- [ ] Exercise 3: Fit model & make predictions
- [ ] Lesson 4: Pahami train/validation split & MAE
- [ ] Exercise 4: Calculate validation MAE
- [ ] Lesson 5: Pahami underfitting vs overfitting
- [ ] Exercise 5: Find optimal `max_leaf_nodes`
- [ ] Lesson 6: Build Random Forest model
- [ ] Exercise 6: Compare Random Forest vs Decision Tree
- [ ] Lesson 7: Submit ke Kaggle competition
- [ ] Exercise 7: Create submission. csv & submit
- [ ] **Bonus:** Improve model (tambah features, tuning)
- [ ] **Bonus:** Download sertifikat & upload LinkedIn

---

## 🎓 Credits & Acknowledgments

**Course Created by:** Kaggle (Dan Becker, Alexis Cook)  
**Dataset:** Ames Housing Dataset (Dean De Cock)  
**Libraries:** scikit-learn, pandas, numpy (open-source community)  
**Documentation Author:** [Your Name / GitHub Profile]  
**Last Updated:** 2025-12-12

---

## 📄 License

This documentation is provided for educational purposes.   
Course content © Kaggle.   
Code examples:  MIT License (free to use & modify).

---

## 🙏 Feedback & Contributions

Found a typo or want to improve this documentation?   
- Open an issue on GitHub
- Submit a pull request
- Contact:  [Your Email / GitHub]

---

**Selamat Belajar!  🚀 Happy Machine Learning!  🎉**

---

## Appendix A:   Glossary

| Term | Definisi |
|------|----------|
| **Algorithm** | Prosedur/formula untuk solve problem (mis.  Decision Tree) |
| **Classification** | Prediksi kategori (mis. spam/not spam) |
| **Cross-Validation** | Teknik validasi dengan multiple train/val splits |
| **Decision Tree** | Model yang membuat keputusan dengan splits berdasarkan features |
| **Ensemble** | Kombinasi multiple models (mis. Random Forest) |
| **Features (X)** | Input variables untuk model |
| **Hyperparameter** | Parameter yang diset sebelum training (mis. `max_depth`) |
| **MAE** | Mean Absolute Error — metrik evaluasi regresi |
| **Model** | Representasi matematis yang dipelajari dari data |
| **Overfitting** | Model terlalu spesifik pada training data → buruk di data baru |
| **Prediction** | Output model untuk input baru |
| **Random Forest** | Ensemble dari banyak Decision Trees |
| **Regression** | Prediksi nilai kontinu (mis. harga rumah) |
| **Supervised Learning** | Learning dari data dengan label/target |
| **Target (y)** | Variable yang ingin diprediksi |
| **Training Data** | Data untuk melatih model |
| **Underfitting** | Model terlalu sederhana → gagal menangkap pola |
| **Validation Data** | Data terpisah untuk evaluasi (tidak dipakai saat training) |

---

## Appendix B:   Common Parameter Values

### DecisionTreeRegressor

```python
DecisionTreeRegressor(
    max_depth=None,           # Unlimited depth (default)
    max_leaf_nodes=None,      # Unlimited leaves (default)
    min_samples_split=2,      # Min samples untuk split (default)
    min_samples_leaf=1,       # Min samples di leaf (default)
    random_state=None         # Set untuk reproducibility
)
```

**Recommended untuk tuning:**
- `max_leaf_nodes`: [50, 100, 250, 500]
- `max_depth`: [5, 10, 15, 20]

### RandomForestRegressor

```python
RandomForestRegressor(
    n_estimators=100,         # Jumlah trees (default: 100)
    max_depth=None,           # Unlimited depth (default)
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='auto',      # Sqrt(n_features) untuk regression
    random_state=None
)
```

**Recommended untuk tuning:**
- `n_estimators`: [100, 200, 300]
- `max_depth`: [10, 15, 20, None]

---

**END OF DOCUMENTATION**

**Version History:**
- v1.0 (2025-12-12) — Initial complete documentation

**Maintainer:** [Your Name]  
**Contact:** [Your GitHub / Email]

🎉 **Congratulations on completing Intro to Machine Learning! ** 🎉


</details>

---

## ✅ License

Dokumentasi ini untuk tujuan edukasi.  
Kode contoh bebas dipakai dan dimodifikasi.

---

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=14&duration=1900&pause=700&color=22C55E&center=true&vCenter=true&width=920&lines=Happy+Learning+%F0%9F%9A%80+Build+your+first+model+and+ship+to+Kaggle!" />

</div>
