# Comprehensive Comparison: Python GNN vs R Traditional Spatial Models
## Indonesian Provincial Inflation Analysis (Jan 2024 - Aug 2025)

---

## Dataset Overview
- **Observations**: 779 total (624 training, 155 testing)
- **Provinces**: 38 Indonesian provinces
- **Time Period**: 20 months (January 2024 - August 2025)
- **Features**: 45 economic indicators
  - Food prices (33 variables: rice, chili, onion, meat, oil, sugar, eggs)
  - Banking data (12 variables: credit, deposits, investments from SEKDA)
- **Target**: Inflasi_MoM (Month-over-Month Inflation based on Consumer Price Index)
- **Split**: Temporal split - 80% train (16 months), 20% test (4 months)

---

## Models Compared

### Python Models (Graph Neural Networks)
1. **OLS Baseline** - Traditional linear regression
2. **GNN-GTVC** - Graph Neural Network with Geographically & Temporally Varying Coefficients
   - Architecture: GCN with 2 layers, 64 hidden dimensions
   - Parameters: 19,451 trainable parameters
   - Spatial weighting: Inverse distance
3. **GNN-GTWR** - Graph Neural Network with Geographically & Temporally Weighted Regression
   - Architecture: GCN with 2 layers, 64 hidden dimensions  
   - Parameters: 24,674 trainable parameters
   - Spatial + Temporal weighting

### R Models (Traditional Statistical Methods)
1. **OLS** - Ordinary Least Squares (baseline)
2. **WLS** - Weighted Least Squares (weighted by residual variance)
3. **GWR-Adaptive** - Geographically Weighted Regression with adaptive kernel
   - Optimal bandwidth: 393 neighbors
   - Bisquare kernel function
4. **GWR-Fixed** - Geographically Weighted Regression with fixed kernel
   - Optimal bandwidth: 33.73 km
   - Bisquare kernel function
5. **GTWR-Adaptive** - ❌ Failed (singular matrix, multicollinearity issues)
6. **GTWR-Fixed** - ❌ Failed (singular matrix, multicollinearity issues)

---

## Performance Comparison

### Test Set Performance (Most Important)

| Model | Platform | Test R² | Test RMSE | Test MAE | Training Time |
|-------|----------|---------|-----------|----------|---------------|
| **WLS** | **R** | **-0.6226** | **0.7706** | **0.5824** | < 1 sec |
| OLS (Python) | Python | -0.7088 | 0.7908 | 0.5971 | < 1 sec |
| OLS (R) | R | -0.7088 | 0.7908 | 0.5971 | < 1 sec |
| GNN-GTVC | Python | -0.6360 | 0.7738 | 0.6089 | ~1 min |
| GWR-Adaptive | R | -0.7909 | 0.8096 | 0.5964 | ~9 sec |
| GNN-GTWR | Python | -0.9234 | 0.8390 | 0.6656 | ~15 min |
| GWR-Fixed | R | -7.5876 | 1.7728 | 0.9066 | ~10 sec |

### Training Set Performance

| Model | Platform | Train R² | Train RMSE | Train MAE |
|-------|----------|----------|------------|-----------|
| **GNN-GTWR** | **Python** | **0.7554** | **0.3522** | **0.3088** |
| GNN-GTVC | Python | 0.7473 | 0.3579 | 0.3128 |
| GWR-Adaptive | R | 0.2710 | 0.6079 | 0.4589 |
| OLS | Both | 0.2163 | 0.6303 | 0.4724 |
| WLS | R | 0.2127 | 0.6318 | 0.4654 |
| GWR-Fixed | R | -5.4427 | 1.8073 | 0.7996 |

---

## Key Findings

### 1. All Models Show Overfitting
- **Negative test R²** across all models indicates predictions worse than simply using the mean
- Large gap between training and testing performance
- Root cause: Limited temporal data (only 20 months, 4 test months)

### 2. Best Performing Model: WLS (R)
- ✅ **Best generalization** among all models
- Test R² = -0.6226 (closest to zero)
- Test RMSE = 0.7706 (lowest among all)
- Weighted approach accounts for heteroskedasticity
- **12.17% improvement** over OLS baseline

### 3. GNN Models Show Severe Overfitting
- Train R² up to 0.76 but Test R² down to -0.92
- GNN-GTWR: 118% overfitting (Train 0.76 → Test -0.92)
- High capacity models (19k-24k parameters) struggle with small dataset
- Training time: 1-15 minutes vs < 10 seconds for R models

### 4. Spatial Heterogeneity Limited Impact
- GWR-Adaptive performs worse than OLS
- Spatial weighting doesn't improve generalization
- Suggests inflation drivers are more global than local in Indonesia

### 5. GTWR Models Failed
- **R implementation**: Singular matrix errors (multicollinearity)
- High-dimensional features (45 features) with limited observations
- Temporal dimension adds complexity that data cannot support

---

## Statistical Significance

### Model Complexity vs Performance

| Model Type | Parameters | Train R² | Test R² | Overfitting Gap |
|------------|-----------|----------|---------|-----------------|
| OLS/WLS | 45 | 0.22 | -0.62 to -0.71 | 0.83-0.93 |
| GWR | Variable | 0.27 | -0.79 | 1.06 |
| GNN-GTVC | 19,451 | 0.75 | -0.64 | 1.39 |
| GNN-GTWR | 24,674 | 0.76 | -0.92 | **1.68** |

**Conclusion**: More parameters → worse overfitting with this dataset size

---

## Spatial Visualization Results

### Interactive Map (Python - Folium)
- Successfully created: `indonesia_inflation_map.html`
- 31 out of 38 provinces visualized
- 639 out of 779 observations with valid geometries
- Inflation range: 0.05% (Gorontalo) to 0.41% (Papua Pegunungan)

### Static Map (Python - Matplotlib)
- File: `indonesia_inflation_static_map.png`
- Shows spatial distribution across Indonesian archipelago
- Highlights eastern provinces with higher inflation

---

## Computational Efficiency

| Model | Training Time | Prediction Time | Scalability |
|-------|--------------|-----------------|-------------|
| OLS | < 1 sec | < 0.1 sec | ✅ Excellent |
| WLS | < 1 sec | < 0.1 sec | ✅ Excellent |
| GWR-Adaptive | ~9 sec | ~1 sec | ✅ Good |
| GWR-Fixed | ~10 sec | ~1 sec | ✅ Good |
| GNN-GTVC | ~60 sec | ~1 sec | ⚠️ Moderate |
| GNN-GTWR | ~900 sec | ~2 sec | ❌ Poor |

---

## Python vs R Comparison

### Python Advantages
1. ✅ Modern deep learning capabilities (PyTorch, PyG)
2. ✅ Flexible neural architecture design
3. ✅ Better visualization (Folium, interactive maps)
4. ✅ Can handle non-linear relationships
5. ✅ GPU acceleration potential

### R Advantages  
1. ✅ **Better performance** on this dataset (WLS best model)
2. ✅ Faster training and inference
3. ✅ Mature spatial statistics packages (GWmodel, sp, sf)
4. ✅ More interpretable results
5. ✅ Lower computational requirements
6. ✅ Proven methods with theoretical foundations

### When to Use Each

**Use Python (GNN):**
- Large datasets (>10,000 observations)
- Complex non-linear relationships
- Rich graph structure (multiple edge types)
- GPU resources available
- Research/experimental settings

**Use R (Traditional):**
- Small to medium datasets (<5,000 observations)
- Need interpretability
- Limited computational resources
- Production deployments
- Policy recommendations

---

## Recommendations

### For This Specific Dataset:
1. ✅ **Use WLS model** for inflation forecasting (best test R²)
2. ❌ **Avoid GNN models** - severe overfitting with current data size
3. ⚠️ **GWR shows potential** but needs more data

### To Improve Model Performance:
1. **Collect more temporal data**: Currently only 20 months
   - Target: At least 36-60 months for reliable time series
   - More test periods (current: only 4 months)

2. **Feature selection**: Reduce 45 features to top 10-15
   - Use LASSO/Ridge regression for feature importance
   - Address multicollinearity issues

3. **Cross-validation**: Implement time series cross-validation
   - Current: Single temporal split
   - Better: Rolling window validation

4. **Ensemble methods**: Combine multiple models
   - Average predictions from OLS, WLS, GWR
   - May improve robustness

5. **Domain knowledge**: Incorporate economic theory
   - Spatial spillover effects between provinces
   - Lag variables (previous month's inflation)
   - External factors (global commodity prices, exchange rates)

---

## Conclusions

### Main Findings:
1. **WLS (R) is the best model** for Indonesian provincial inflation
   - Simple, fast, interpretable, best generalization
   
2. **GNN models overfit severely** due to:
   - Small dataset (779 observations)
   - High model capacity (19k-24k parameters)
   - Limited temporal observations (20 months)
   
3. **Spatial heterogeneity is limited**:
   - Local spatial effects don't improve predictions
   - Inflation drivers appear more global/national
   
4. **Data quantity is the bottleneck**:
   - Need 3-5x more temporal observations
   - Current 20 months insufficient for complex models

### Practical Implications:
- **For policy makers**: Use simple WLS model for short-term forecasts
- **For researchers**: Collect more data before attempting advanced methods
- **For developers**: R traditional methods preferred for production systems

### Future Work:
1. Extend data collection to 36-60 months
2. Include external variables (oil prices, USD-IDR exchange rate, global inflation)
3. Test ensemble methods combining OLS, WLS, and GWR
4. Investigate province-specific time series models (ARIMA, VAR)
5. Collect higher-frequency data (weekly instead of monthly)

---

## Output Files Generated

### Python Analysis:
- `GNN_GTVC_GTWR_Implementation copy 2.ipynb` - Complete analysis notebook
- `GNN_GTVC_GTWR_Comprehensive_Results.csv` - All model results
- `indonesia_inflation_map.html` - Interactive Folium map
- `indonesia_inflation_static_map.png` - Static matplotlib map
- `experimental_methods_comparison.png` - Model comparison chart

### R Analysis:
- `Models in R.ipynb` - Complete R analysis notebook
- `R_Models_Comparison_Results.csv` - R model comparison results
- `R_Models_Performance_Comparison.png` - Visualization of R models

### This Report:
- `Python_R_Models_Comparison.md` - Comprehensive comparison document

---

## References

### Python Packages Used:
- PyTorch 2.3.1+cpu
- PyTorch Geometric 2.3.1
- GeoPandas 0.14.3
- Folium 0.15.1
- Scikit-learn 1.6.1

### R Packages Used:
- GWmodel 2.3-1 (Geographically Weighted Regression)
- sp 1.6-0 (Spatial objects)
- sf 1.0-12 (Simple features)
- spdep 1.2-8 (Spatial dependence)
- dplyr 1.1.4, ggplot2 3.4.4

---

**Analysis Date**: 2025
**Dataset Period**: January 2024 - August 2025
**Location**: Indonesia (38 Provinces)
**Target Variable**: Month-over-Month Inflation (from Consumer Price Index)

---

*This analysis demonstrates that for small temporal datasets, traditional statistical methods (WLS, OLS) outperform complex deep learning approaches (GNN). Data quantity, not model sophistication, is the limiting factor.*
