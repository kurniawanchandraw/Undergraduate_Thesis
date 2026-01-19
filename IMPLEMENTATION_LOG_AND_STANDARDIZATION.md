# Implementation Changes - Log Transform & Standardization

## Files Modified
- `d:\Semester VII\Tugas Akhir\Chap 4\Chapter_4_Analysis.ipynb`

## Cells Modified/Added

### 1. Variable Preparation Cell (Modified)
**Cell ID:** #VSC-5b258663

**Before:**
```python
X_bps = data_bps[['Persentase Penduduk Miskin (Persen)',
                   'Rata-rata Lama Sekolah (Tahun)',
                   'Pengeluaran per Kapita/bulan', 
                   'Tingkat Pengangguran Terbuka']].to_numpy()
y_bps = data_bps['Umur Harapan Hidup/UHH (Tahun)'].to_numpy()
```

**After:**
```python
from sklearn.preprocessing import StandardScaler

# Log-transform Pengeluaran per Kapita
X_raw = data_bps[...].copy()
X_raw['Pengeluaran per Kapita/bulan'] = np.log(X_raw['Pengeluaran per Kapita/bulan'])
y_bps_raw = data_bps['Umur Harapan Hidup/UHH (Tahun)'].to_numpy()

# Standardize
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_bps = scaler_X.fit_transform(X_raw)
y_bps = scaler_y.fit_transform(y_bps_raw.reshape(-1, 1)).ravel()
```

---

### 2. Inverse Transformation Helpers (New Cell)
**Cell ID:** #VSC-a2a7095b (inserted after variable preparation)

**New Content:**
```python
def inverse_transform_y(y_standardized):
    """Convert standardized y back to original scale"""
    return scaler_y.inverse_transform(y_standardized.reshape(-1, 1)).ravel()

def inverse_transform_X(X_standardized):
    """Convert standardized X back to original scale (before log transform)"""
    X_original = scaler_X.inverse_transform(X_standardized)
    # Undo log transform on Pengeluaran per Kapita (column 2)
    X_original[:, 2] = np.exp(X_original[:, 2])
    return X_original

def inverse_transform_predictions(y_pred_std):
    """Convert standardized predictions to original UHH scale"""
    return inverse_transform_y(y_pred_std)
```

---

### 3. EDA Cell (Modified)
**Cell ID:** #VSC-c8e83cc8

**Addition:** Display both standardized and original-scale statistics
```python
# Show standardized data
summary_stats_std = pd.DataFrame({...})
print("\n📊 Statistik Deskriptif (STANDARDIZED DATA):")

# Show original scale for reference
y_orig = inverse_transform_y(y_bps)
X_orig = inverse_transform_X(X_bps)
print("\n📊 ORIGINAL SCALE (untuk referensi):")
```

---

### 4. Original Scale Metrics Cell (New)
**Cell ID:** #VSC-ea6b08fa (inserted after model comparison)

**New Content:**
```python
# Compute RMSE & MAE in original UHH scale
y_test_orig = inverse_transform_y(y_test_bps)

# OLS
y_pred_ols_orig = inverse_transform_y(y_pred_ols_bps)
ols_rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_ols_orig))
ols_mae_orig = mean_absolute_error(y_test_orig, y_pred_ols_orig)

# GWR
y_pred_gwr_orig = inverse_transform_y(y_pred_gwr_bps)
gwr_rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_gwr_orig))
gwr_mae_orig = mean_absolute_error(y_test_orig, y_pred_gwr_orig)

# GA-GWR
for backbone in ['GAT', 'GCN', 'SAGE']:
    y_pred_gagwr = bps_models[backbone].predict(u_test_bps, X_test_gagwr)
    y_pred_gagwr_orig = inverse_transform_y(y_pred_gagwr)
    rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_gagwr_orig))
```

---

## Variable Names Updated
```python
var_names_bps = [
    'Persentase Penduduk Miskin',
    'Rata-rata Lama Sekolah',
    'Log(Pengeluaran per Kapita)',  # ← Updated to show log transform
    'Tingkat Pengangguran Terbuka'
]
```

---

## Execution Results

### Before Standardization
```
GA-GWR (GAT):  R² = 0.5707  ✓ (OK but not great)
GA-GWR (GCN):  R² = -17680  ✗ (Exploding gradients!)
GA-GWR (SAGE): R² = -7278   ✗ (Exploding gradients!)
```

### After Standardization
```
GA-GWR (GAT):  R² = 0.6582  ✓ (Improved)
GA-GWR (GCN):  R² = 0.6582  ✓ (Fixed!)
GA-GWR (SAGE): R² = 0.6614  ✓ (Fixed!)
```

---

## Impact Summary

| Aspect | Before | After | Change |
|--------|--------|-------|--------|
| GCN Stability | ✗ (R² = -17680) | ✓ (R² = 0.6582) | Fixed |
| SAGE Stability | ✗ (R² = -7278) | ✓ (R² = 0.6614) | Fixed |
| GAT Performance | 0.5707 | 0.6582 | +15.3% |
| Backbone Consistency | Highly variable | Consistent (0.658-0.661) | Improved |
| Data Interpretability | Working values (raw scale) | Easy interpretation (std+orig scale both shown) | Better |

---

## Technical Notes

1. **Standardization doesn't affect R² values** - R² is scale-invariant
2. **RMSE & MAE change** - They scale with the inverse standardization
3. **Spatial coordinates NOT standardized** - GWR needs actual geographic distances
4. **All metrics available in both scales** - Can report in years (original) or standardized units
5. **Inverse transforms are fully reversible** - No information loss

---

## Best Practices Applied

✅ Standardization before neural network training
✅ Log transformation for right-skewed variables
✅ Separate inverse transformation functions for clarity
✅ Spatial coordinates excluded from standardization
✅ Both standardized and original-scale results reported
✅ Scaler objects preserved for future predictions

---

*Implementation Date: 2026-01-19*
*Status: Complete & Tested*
