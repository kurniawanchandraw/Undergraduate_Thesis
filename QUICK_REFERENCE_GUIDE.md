# 🚀 Quick Reference: Standardization & Log Transform Implementation

## TL;DR - What Was Done

```python
# 1. LOG TRANSFORM
X_raw[:, 2] = np.log(X_raw[:, 2])  # Pengeluaran per Kapita

# 2. STANDARDIZE
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_bps = scaler_X.fit_transform(X_raw)
y_bps = scaler_y.fit_transform(y_bps_raw.reshape(-1,1)).ravel()

# 3. USE STANDARDIZED DATA FOR MODELS
# All models trained on X_bps, y_bps

# 4. INVERSE TRANSFORM FOR RESULTS
y_pred_original = scaler_y.inverse_transform(y_pred_std.reshape(-1,1)).ravel()
```

---

## Impact Summary

| Aspect | Before | After |
|--------|--------|-------|
| **GCN** | Explodes (R² = -17,680) | Works (R² = 0.6582) ✅ |
| **SAGE** | Explodes (R² = -7,278) | Works (R² = 0.6614) ✅ |
| **GAT** | Okay (R² = 0.5707) | Better (R² = 0.6582) ↑ 15% |
| **Consistency** | Highly variable | Consistent ✅ |

---

## Using the Models Going Forward

### For New Predictions

```python
# Load fitted scalers (saved in notebook kernel)
# X_new_raw = get new data

# 1. Apply log transform
X_new[:, 2] = np.log(X_new[:, 2])

# 2. Apply standardization (use fitted scalers!)
X_new_std = scaler_X.transform(X_new)  # transform, NOT fit_transform!

# 3. Get predictions (in standardized scale)
y_pred_std = model.predict(u_new, X_new_std)

# 4. Inverse transform to original scale
y_pred_original = scaler_y.inverse_transform(y_pred_std.reshape(-1,1)).ravel()
```

### Important: Use Fitted Scalers

```python
# ✅ CORRECT - Uses scalers fitted on original training data
X_new_std = scaler_X.transform(X_new)

# ❌ WRONG - Would refit on new data, losing consistency
X_new_std = scaler_X.fit_transform(X_new)
```

---

## Interpretation Guide

### Reading Standardized Values
```
y_std = 1.5 means: 1.5 standard deviations above mean UHH
y_original = scaler_y.inverse_transform([[1.5]])
           = 72.07 + (1.5 × 2.90)
           = 76.42 years
```

### Reading Original Values
```
y_original = 72.5 years UHH
y_std = (72.5 - 72.07) / 2.90 = 0.148
```

---

## Scaler Parameters (Reference)

```python
scaler_X.mean_ = [11.8765, 8.4400, 13.9499, 4.7876]
scaler_X.scale_ = [7.4238, 1.6354, 0.2883, 2.4752]

scaler_y.mean_ = 72.0729
scaler_y.scale_ = 2.8975

# Variable order:
# 0: Persentase Penduduk Miskin
# 1: Rata-rata Lama Sekolah
# 2: Log(Pengeluaran per Kapita)  ← LOG TRANSFORMED!
# 3: Tingkat Pengangguran Terbuka
```

---

## Common Mistakes to Avoid

❌ **Don't refit scaler on test/new data**
```python
# WRONG:
X_test_std = scaler_X.fit_transform(X_test)
# Creates different standardization for test set!
```

✅ **Do transform using fitted scaler**
```python
# CORRECT:
X_test_std = scaler_X.transform(X_test)
# Uses same standardization as training
```

---

❌ **Don't forget log transform on Pengeluaran per Kapita**
```python
# WRONG:
X_new = data_new[['Pengeluaran per Kapita/bulan', ...]].values
X_new_std = scaler_X.transform(X_new)
# Mismatch! Scaler expects log-transformed values
```

✅ **Do apply log transform first**
```python
# CORRECT:
X_new = data_new[['Pengeluaran per Kapita/bulan', ...]].copy()
X_new['Pengeluaran per Kapita/bulan'] = np.log(X_new['Pengeluaran per Kapita/bulan'])
X_new_std = scaler_X.transform(X_new)
```

---

❌ **Don't standardize spatial coordinates**
```python
# WRONG:
u_std = scaler_u.transform(u)  # Breaks GWR spatial calculations!
gwr_model.fit(u_std, X, y)
```

✅ **Keep spatial coordinates in original scale**
```python
# CORRECT:
gwr_model.fit(u, X_std, y_std)  # u in meters/degrees, X & y standardized
```

---

## File Locations

### Notebook
📓 `Chapter_4_Analysis.ipynb` - Cells 32-54 contain BPS analysis with standardization

### Visualizations (8 PDFs)
📁 `Chap 4\figures\`
- BPS_01_Coefficient_Maps.pdf
- BPS_04_Significance_Maps.pdf
- BPS_05_Cluster_Map.pdf
- BPS_06_Residual_Diagnostics.pdf
- BPS_07_Coefficient_Boxplots.pdf
- BPS_08_Model_Comparison_Stable.pdf

### Documentation
📄 `STANDARDIZATION_SUMMARY.md` - Overview
📄 `IMPLEMENTATION_LOG_AND_STANDARDIZATION.md` - Technical details
📄 `TECHNICAL_BEFORE_AFTER_ANALYSIS.md` - Detailed comparison
📄 `BPS_ANALYSIS_COMPLETION_REPORT.md` - Final report

---

## Quick Model Comparison

```
Model           R²      RMSE (std)  MAE (std)   RMSE (years)  MAE (years)
──────────────────────────────────────────────────────────────────────
OLS            0.308   0.6465      0.5267      1.8731        1.5262
GA-GWR (GAT)   0.658   0.4544      0.3423      1.3168        0.9918
GA-GWR (GCN)   0.658   0.4544      0.3423      1.3167        0.9917
GA-GWR (SAGE)  0.661   0.4523      0.3401      1.3105        0.9855
Classical GWR  0.737   0.3989      0.3262      1.1557        0.9450
```

**Best Model:** Classical GWR (R² = 0.7367)

---

## Batch Processing Template

For processing multiple new observations:

```python
import numpy as np
import pandas as pd

def predict_uhh_new(data_new):
    """
    Predict UHH for new data
    
    Parameters:
    -----------
    data_new : DataFrame with columns
        - longitude, latitude
        - Persentase Penduduk Miskin (Persen)
        - Rata-rata Lama Sekolah (Tahun)
        - Pengeluaran per Kapita/bulan
        - Tingkat Pengangguran Terbuka
    
    Returns:
    --------
    DataFrame with predictions in original scale (UHH years)
    """
    
    # Extract coordinates (not standardized)
    u_new = data_new[['longitude', 'latitude']].values
    
    # Extract & transform X
    X_new = data_new[['Persentase Penduduk Miskin (Persen)',
                       'Rata-rata Lama Sekolah (Tahun)',
                       'Pengeluaran per Kapita/bulan',
                       'Tingkat Pengangguran Terbuka']].copy()
    X_new['Pengeluaran per Kapita/bulan'] = np.log(X_new['Pengeluaran per Kapita/bulan'])
    
    # Standardize using fitted scalers
    X_new_std = scaler_X.transform(X_new)
    
    # Get predictions (standardized)
    y_pred_std = best_model.predict(u_new, X_new_std)
    
    # Inverse transform to original scale
    y_pred_original = scaler_y.inverse_transform(y_pred_std.reshape(-1,1)).ravel()
    
    # Return results
    return pd.DataFrame({
        'Prediksi_UHH': y_pred_original,
        'Confidence': 'Based on 0.737 R² model'
    })

# Usage:
# results = predict_uhh_new(new_data)
```

---

## Troubleshooting

### Problem: "AssertionError: X contains NaN, infinity or a value too large"
**Cause:** Forgot log transform on Pengeluaran per Kapita  
**Solution:** Add `X['Pengeluaran per Kapita'] = np.log(X['Pengeluaran per Kapita'])`

### Problem: "ValueError: X has n_features=4 but this estimator was fitted with n_features=5"
**Cause:** Forgot intercept column or dimension mismatch  
**Solution:** Check input shape matches training data (after all transforms)

### Problem: Predictions seem way off (e.g., 500 years UHH)
**Cause:** Forgot to inverse transform predictions  
**Solution:** Add `y_pred_original = scaler_y.inverse_transform(y_pred_std.reshape(-1,1)).ravel()`

### Problem: Model works on training data but fails on new data
**Cause:** Applied log to DIFFERENT column or forgot log entirely  
**Solution:** Always apply transformations in exact same order:
   1. Load raw data
   2. Apply log to Pengeluaran per Kapita
   3. Apply standardization using fitted scalers

---

## Key Takeaways

1. ✅ **Standardization is non-negotiable for neural networks**
2. ✅ **Log-transform for right-skewed features**
3. ✅ **Keep spatial coordinates in original scale**
4. ✅ **Always use fitted scaler for new data** (transform, not fit_transform)
5. ✅ **Report results in both standardized AND original scales**
6. ✅ **Save scaler objects with trained models**

---

**Status:** ✅ Ready for Production Use  
**Version:** 1.0  
**Date:** 2026-01-19
