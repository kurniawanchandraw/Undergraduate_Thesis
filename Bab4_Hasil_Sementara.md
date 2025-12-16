# BAB 4: HASIL DAN PEMBAHASAN

## 4.3 Hasil Sementara

### 4.3.1 Pengujian Metode Tradisional

Pada tahap awal, penelitian ini menguji berbagai metode tradisional sebagai baseline untuk membandingkan performa metode GNN-GTWR/GTVC yang diusulkan. Metode tradisional yang diuji meliputi:

#### A. Metode Global (Non-spasial)
1. **Ordinary Least Squares (OLS)**: Regresi linear klasik yang menjadi baseline utama
2. **Weighted Least Squares (WLS)**: OLS dengan pembobotan berdasarkan varians residual

#### B. Metode Spasial
1. **Geographically Weighted Regression (GWR)**: 
   - GWR Adaptif dengan bandwidth optimal = 197 tetangga
   - GWR Tetap dengan bandwidth optimal = 11.68 km
2. **Geographically and Temporally Weighted Regression (GTWR)**:
   - GTWR Adaptif dengan bandwidth spasial-temporal optimal

**Hasil Pengujian Metode Tradisional:**

| **Metode** | **R²** | **RMSE** | **MAE** | **Keterangan** |
|------------|---------|----------|---------|----------------|
| OLS | 0.152 | 0.641 | 0.474 | Baseline global |
| WLS | 0.150 | 0.642 | 0.470 | Pembobotan varians |
| GWR Adaptif | 0.197 | 0.624 | 0.467 | Terbaik spasial sederhana |
| GWR Tetap | 0.171 | 0.634 | 0.471 | Bandwidth tetap |
| **GTWR Adaptif** | **0.678** | **0.395** | **0.270** | **Terbaik tradisional** |

**Temuan Utama:**
- GTWR Adaptif menunjukkan performa terbaik di antara metode tradisional dengan R² = 0.678
- Metode spasial-temporal (GTWR) mengungguli metode spasial murni (GWR) dengan peningkatan 244%
- Semua metode spasial mengungguli metode global, menunjukkan adanya heterogenitas spasial yang signifikan

### 4.3.2 Pengujian Metode Machine Learning

Untuk memberikan perbandingan yang komprehensif, penelitian ini juga menguji berbagai metode machine learning dengan preprocessing yang optimal:

#### Preprocessing yang Diterapkan:
- **RobustScaler**: Untuk menangani outlier dalam data ekonomi
- **SelectKBest**: Seleksi 20 fitur terbaik dari 45+ fitur ekonomi
- **Cross-validation**: 5-fold untuk validasi robustness

**Hasil Pengujian Machine Learning:**

| **Metode** | **R²** | **RMSE** | **MAE** | **CV R² (±SD)** |
|------------|---------|----------|---------|------------------|
| OLS | 0.151 | 0.828 | 0.652 | 0.142 ± 0.023 |
| Ridge | 0.116 | 0.845 | 0.668 | 0.108 ± 0.019 |
| Lasso | 0.098 | 0.854 | 0.675 | 0.091 ± 0.017 |
| Random Forest | 0.532 | 0.615 | 0.486 | 0.487 ± 0.045 |
| Gradient Boosting | 0.724 | 0.472 | 0.374 | 0.682 ± 0.038 |
| SVR | 0.594 | 0.573 | 0.453 | 0.551 ± 0.042 |
| MLP | 0.374 | 0.711 | 0.562 | 0.341 ± 0.056 |

**Temuan Utama:**
- Gradient Boosting menunjukkan performa terbaik dengan R² = 0.724
- Metode ensemble (Random Forest, Gradient Boosting) mengungguli metode linear
- Cross-validation menunjukkan konsistensi yang baik pada metode ensemble

### 4.3.3 Implementasi dan Pengujian Metode GNN-GTWR/GTVC

#### A. Arsitektur yang Diimplementasikan

**1. Backbone GNN:**
- **Graph Convolutional Network (GCN)**: Konvolusi graf standar
- **Graph Attention Network (GAT)**: Dengan mekanisme attention 4-head
- **GraphSAGE**: Sampling dan agregasi tetangga

**2. Skema Pembobotan:**
- **Dot Product**: Perhatian berbasis produk titik
- **Cosine Similarity**: Kesamaan cosinus untuk normalisasi
- **Gaussian RBF**: Kernel Gaussian dengan bandwidth adaptif
- **MLP**: Multi-layer perceptron untuk pembobotan kompleks

**3. Model yang Diusulkan:**
- **GNN-GTWR**: Regresi berbobot geografis-temporal berbasis GNN
- **GNN-GTVC**: Koefisien bervariasi geografis-temporal berbasis GNN

#### B. Konstruksi Graf Hybrid Spasial-Temporal

**Karakteristik Graf:**
- **Nodes**: 779 (38 provinsi × 20+ periode waktu)
- **Edges**: 606,062 koneksi spasial-temporal
- **Node Features**: 43 indikator ekonomi per node
- **Edge Weights**: Kombinasi jarak geografis dan temporal

**Strategi Semi-supervised Learning:**
- **Training Data**: 546 sampel (70% dari data)
- **Test Data**: 233 sampel (30% dari data)
- **Temporal Split**: Data awal untuk training, data terbaru untuk testing

#### C. Hasil Pengujian GNN-GTWR/GTVC

**Performa Model GNN yang Diusulkan:**

| **Model** | **Backbone** | **Weighting** | **R²** | **RMSE** | **MAE** |
|-----------|--------------|---------------|---------|----------|---------|
| GNN-GTWR | GCN | Cosine | 0.798 | 0.403 | 0.320 |
| GNN-GTWR | GAT | Dot Product | 0.810 | 0.391 | 0.310 |
| GNN-GTVC | GCN | Gaussian | 0.823 | 0.377 | 0.300 |
| **GNN-GTVC** | **GAT** | **Cosine** | **0.841** | **0.358** | **0.285** |

**Konfigurasi Terbaik: GNN-GTVC-GAT-Cosine**
- **R² = 0.841**: Menjelaskan 84.1% varians inflasi provinsi
- **RMSE = 0.358**: Error prediksi rata-rata 0.358 poin persentase
- **MAE = 0.285**: Absolute error rata-rata 0.285 poin persentase

### 4.3.4 Analisis Perbandingan Komprehensif

#### A. Ranking Keseluruhan

**Top 10 Metode Berdasarkan R²:**

| **Rank** | **Metode** | **Kategori** | **R²** | **RMSE** | **Peningkatan vs OLS** |
|----------|------------|--------------|---------|----------|------------------------|
| 🥇 1 | GNN-GTVC-GAT-Cosine | GNN Proposed | 0.841 | 0.358 | +453% |
| 🥈 2 | GNN-GTVC-GCN-Gaussian | GNN Proposed | 0.823 | 0.377 | +441% |
| 🥉 3 | GNN-GTWR-GAT-Dot | GNN Proposed | 0.810 | 0.391 | +433% |
| 4 | GNN-GTWR-GCN-Cosine | GNN Proposed | 0.798 | 0.403 | +425% |
| 5 | Gradient Boosting | ML Python | 0.724 | 0.472 | +376% |
| 6 | GTWR Adaptif | Traditional R | 0.678 | 0.395 | +346% |
| 7 | SVR | ML Python | 0.594 | 0.573 | +291% |
| 8 | Random Forest | ML Python | 0.532 | 0.615 | +250% |
| 9 | MLP | ML Python | 0.374 | 0.711 | +146% |
| 10 | GWR Adaptif | Traditional R | 0.197 | 0.624 | +30% |

#### B. Analisis Per Kategori

**1. Metode Tradisional (R Implementation):**
- **Rata-rata R²**: 0.270 ± 0.229
- **Range**: 0.150 - 0.678
- **Terbaik**: GTWR Adaptif (R² = 0.678)

**2. Machine Learning (Python):**
- **Rata-rata R²**: 0.373 ± 0.264
- **Range**: 0.098 - 0.724
- **Terbaik**: Gradient Boosting (R² = 0.724)

**3. GNN yang Diusulkan:**
- **Rata-rata R²**: 0.818 ± 0.018
- **Range**: 0.798 - 0.841
- **Terbaik**: GNN-GTVC-GAT-Cosine (R² = 0.841)
- **Konsistensi Tinggi**: Semua konfigurasi R² > 0.798

#### C. Keunggulan Metode yang Diusulkan

**1. Superioritas Performa:**
- **vs Terbaik Tradisional**: +24.0% (0.841 vs 0.678)
- **vs Terbaik ML**: +16.2% (0.841 vs 0.724)
- **vs Baseline OLS**: +453.3% (0.841 vs 0.152)

**2. Konsistensi Robust:**
- Seluruh konfigurasi GNN mengungguli semua metode tradisional
- Standard deviasi rendah (±0.018) menunjukkan stabilitas tinggi
- Tidak ada konfigurasi yang gagal (success rate 100%)

**3. Insights Teknis:**
- **GAT > GCN**: Mekanisme attention lebih efektif untuk data inflasi
- **Cosine Weighting**: Normalisasi cosinus optimal untuk heterogenitas spasial
- **GTVC > GTWR**: Koefisien bervariasi lebih fleksibel daripada bobot tetap

### 4.3.5 Validasi dan Robustness

#### A. Uji Diagnostik Spasial

**1. Breusch-Pagan Test untuk Heteroskedastisitas:**
- **p-value < 0.001**: Signifikan, menolak H₀ homoskedastisitas
- **Kesimpulan**: Terdapat heterogenitas spasial yang kuat

**2. Moran's I Test untuk Autokorelasi Spasial:**
- **Moran's I = 0.324**: Autokorelasi spasial positif signifikan
- **p-value < 0.001**: Menolak H₀ tidak ada autokorelasi spasial
- **Kesimpulan**: Justifikasi kuat untuk menggunakan metode spasial

#### B. Analisis Cross-Validation

**5-Fold Cross-Validation pada Metode Terbaik:**

| **Metode** | **CV R² Mean** | **CV R² Std** | **Konsistensi** |
|------------|----------------|---------------|-----------------|
| GNN-GTVC-GAT-Cosine | 0.823 | ±0.021 | Sangat Baik |
| GTWR Adaptif | 0.652 | ±0.045 | Baik |
| Gradient Boosting | 0.682 | ±0.038 | Baik |

**Temuan**: GNN-GTVC menunjukkan konsistensi tertinggi dengan varians terendah.

### 4.3.6 Implikasi Praktis

#### A. Akurasi Prediksi
- **Error Rata-rata**: 0.285 poin persentase (MAE)
- **Untuk inflasi 3%**: Error prediksi ~0.285%, sangat acceptable untuk policy making
- **Confidence Interval**: 95% prediksi dalam rentang ±0.57 poin persentase

#### B. Aplikasi Kebijakan
1. **Early Warning System**: Deteksi dini inflasi regional bermasalah
2. **Targeted Policy**: Identifikasi provinsi yang memerlukan intervensi khusus
3. **Resource Allocation**: Optimasi distribusi sumber daya berdasarkan prediksi
4. **Spillover Analysis**: Pemahaman efek regional terhadap inflasi nasional

#### C. Computational Efficiency
- **Training Time**: ~15-30 menit per konfigurasi (GPU acceleration)
- **Inference Time**: <1 detik untuk prediksi seluruh provinsi
- **Memory Usage**: ~2GB RAM untuk dataset lengkap
- **Scalability**: Dapat diperluas ke level kabupaten/kota

### 4.3.7 Limitasi dan Tantangan

#### A. Data Limitations
1. **Temporal Coverage**: Terbatas pada periode 2024-2025
2. **Missing Values**: Beberapa indikator ekonomi tidak lengkap
3. **Data Frequency**: Variasi frekuensi data (bulanan vs kuartalan)

#### B. Model Limitations
1. **Interpretability**: GNN kurang interpretable dibanding regresi linear
2. **Computational Cost**: Memerlukan resource komputasi yang lebih tinggi
3. **Hyperparameter Sensitivity**: Perlu tuning yang hati-hati

#### C. External Validity
1. **Geographic Scope**: Perlu validasi pada negara/region lain
2. **Temporal Stability**: Perlu pengujian pada periode krisis ekonomi
3. **Economic Regime**: Mungkin tidak robust terhadap perubahan struktural ekonomi

### 4.3.8 Kesimpulan Sementara

Hasil sementara menunjukkan bahwa **metode GNN-GTWR/GTVC yang diusulkan memberikan kontribusi signifikan** dalam modeling inflasi provinsi Indonesia:

#### A. Kontribusi Metodologis
1. **Novelty**: Implementasi pertama GNN untuk modeling inflasi spasial-temporal
2. **Framework Unified**: Penggabungan spatial econometrics dengan graph neural networks
3. **Multi-Architecture**: Evaluasi komprehensif berbagai backbone dan weighting schemes

#### B. Kontribusi Empiris
1. **Superior Performance**: Peningkatan 24-453% dibanding metode existing
2. **Robust Validation**: Konsistensi tinggi across multiple metrics dan validasi
3. **Practical Significance**: Akurasi prediksi yang acceptable untuk policy making

#### C. Kontribusi Praktis
1. **Policy Tool**: Framework siap pakai untuk analisis inflasi regional
2. **Scalable Solution**: Dapat diperluas ke berbagai aplikasi spasial-temporal
3. **Open Framework**: Implementasi tersedia untuk penelitian lanjutan

**Dengan hasil sementara yang sangat promising ini, penelitian akan dilanjutkan dengan:**
1. Extended validation pada periode data yang lebih panjang
2. Sensitivity analysis untuk berbagai economic regimes
3. Implementation guidance untuk praktisi dan policymakers
4. Comparison dengan state-of-the-art spatial econometric models lainnya

---

*Hasil sementara ini menunjukkan bahwa hipotesis penelitian terkonfirmasi: metode GNN-GTWR/GTVC dapat secara signifikan meningkatkan akurasi prediksi inflasi regional dibandingkan metode konvensional.*