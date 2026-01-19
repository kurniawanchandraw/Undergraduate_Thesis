# BPS Data Analysis - Standardization & Log Transform Summary

## Changes Applied

### 1. **Log Transformation**
- **Variable**: Pengeluaran per Kapita (per capita spending)
- **Reason**: To normalize right-skewed distribution and improve model stability
- **Implementation**: `log(Pengeluaran per Kapita)` applied before standardization
- **Note**: Automatically inverse-transformed when reporting results in original scale

### 2. **Standardization (Z-score normalization)**
Applied to both **X variables** and **y response variable**:

**X variables (4 predictors):**
```
X_standardized = (X_raw - mean) / std
Results: mean ≈ 0, std = 1
```

**y variable (UHH - Umur Harapan Hidup):**
```
y_standardized = (y_raw - mean) / std
Results: mean ≈ 0, std = 1
```

**Spatial coordinates (u) - NOT standardized:**
- Longitude & Latitude kept in original geographic scale
- This is appropriate since GWR uses actual spatial distances

### 3. **Scaler Parameters Preserved**
For inverse transformation:
```python
scaler_X.mean_ = [11.8765, 8.4400, 13.9499, 4.7876]
scaler_X.scale_ = [7.4238, 1.6354, 0.2883, 2.4752]
scaler_y.mean_ = 72.0729 (UHH average)
scaler_y.scale_ = 2.8975 (UHH std)
```

## Why This Matters for Neural Networks

### Problem Before Standardization
- **GCN & SAGE models** had R² = -17680 and -7278 (exploding gradients)
- Gradient descent became unstable with raw-scale data
- Learning rates and weight initialization couldn't balance across different feature scales
- Pengeluaran per Kapita values (millions) dominated optimization

### Solution with Standardization
✅ **All three backbones now work stably:**
- GAT: R² = 0.6582 (consistent)
- GCN: R² = 0.6582 (now stable!)
- SAGE: R² = 0.6614 (now stable!)

✅ **Improved neural network behavior:**
- Gradients remain in reasonable range
- Weight initialization more effective
- Convergence more reliable
- All backbones perform comparably

## Model Performance Comparison

### Standardized Scale (Training Space)
```
Model                 R²        RMSE      MAE
─────────────────────────────────────────────
Classical GWR       0.7367    0.3989    0.3262
GA-GWR (SAGE)       0.6614    0.4523    0.3401
GA-GWR (GCN)        0.6582    0.4544    0.3423
GA-GWR (GAT)        0.6582    0.4544    0.3423
OLS                 0.3083    0.6465    0.5267
```

### Original Scale (UHH in years)
```
Model                 RMSE         MAE
──────────────────────────────────────
Classical GWR       1.1557 tahun    0.9450 tahun
GA-GWR (SAGE)       1.3105 tahun    0.9855 tahun
GA-GWR (GCN)        1.3167 tahun    0.9917 tahun
GA-GWR (GAT)        1.3168 tahun    0.9918 tahun
OLS                 1.8731 tahun    1.5262 tahun
```

**Interpretation:**
- Classical GWR: Average error of ±1.16 years in UHH prediction
- GA-GWR models: Average error of ±1.31 years
- R² is scale-invariant, so values are the same in both scales

## Inverse Transformation Process

### For Predictions
```python
def inverse_transform_predictions(y_pred_std):
    """Convert standardized predictions back to original UHH scale"""
    return scaler_y.inverse_transform(y_pred_std.reshape(-1, 1)).ravel()
```

### For Pengeluaran per Kapita
```python
def inverse_transform_X(X_standardized):
    """Undo standardization + undo log transform"""
    X_original = scaler_X.inverse_transform(X_standardized)
    # Undo log on column 2 (Pengeluaran per Kapita)
    X_original[:, 2] = np.exp(X_original[:, 2])
    return X_original
```

## Diagnostics (Standardized Data)
```
Test                    Result              p-value
─────────────────────────────────────────────────────
Normality (K-S)         Not Normal          p < 0.001
Heteroscedasticity      Present             p < 0.001
Spatial Autocorr (MI)   NaN (insufficient)  p = NaN
```

## Spatial Clustering Analysis
```
Optimal K: 2
Silhouette Score: 0.8934 (excellent separation)

Cluster 1: 2,048 observations (regular regions)
Cluster 2:     8 observations (outlier regions)
```

## Key Insights

1. **Neural networks are sensitive to feature scales** - standardization was critical for stability
2. **All three GNN architectures now work** - GCN and SAGE are no longer unstable
3. **Classical GWR remains the best model** - R² = 0.7367 beats GA-GWR approaches on this dataset
4. **Spatial heterogeneity is present** - K-means finds meaningful clusters (Silhouette = 0.89)
5. **Spatial weights matter** - GWR outperforms global OLS (R² = 0.73 vs 0.31)

## Data Information
- **Dataset**: BPS Regional Economic Data (Umur Harapan Hidup)
- **Observations**: 2,570 (514 locations × 5 years)
- **Variables**: 4 predictors + 1 response
- **Time Range**: 2019-2023
- **Spatial Scale**: Indonesia (all provinces & districts)
- **Train/Test Split**: 2019-2022 training (80%), 2023 testing (20%)

---
*Generated: 2026-01-19*
*Notebook: Chapter_4_Analysis.ipynb*
