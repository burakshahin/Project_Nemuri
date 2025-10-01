# Project Nemuri - Two Descriptor logP Modeling

**Minimal, Colab-ready repository** for predicting experimental logP from BAR calculations using 2 optimal descriptors.

## 🎯 Goal
Predict experimental logP values from computational BAR calculations using **≤2 descriptors** with maximum interpretability.

## 📊 Current Performance
- **Descriptor 1:** BAR_logP (computational baseline)
- **Descriptor 2:** PolarityProxy = |BAR_Water - BAR_Octanol| / max(diff)
- **Results:** R² = 0.813 (10-fold CV), RMSE = 0.521, MAE = 0.404
- **Improvement over baseline:** +1.0% R² vs simple y = 0.8645x - 0.1688

## 📁 Files
- `logP_calculation.ods` - Source dataset (123 compounds)
- `read_ods_to_csv.py` - ODS→CSV converter
- `logP_data_from_ods.csv` - Clean CSV (auto-generated)
- `model_two_descriptor.py` - 2-descriptor modeling with LinearRegression
- `requirements.txt` - Python dependencies
- `tpot/` - Vendored TPOT library (for future AutoML experiments)

## 🚀 Google Colab Workflow

```python
# 1. Install dependencies
!pip install -r requirements.txt

# 2. Convert ODS to CSV (creates logP_data_from_ods.csv)
!python read_ods_to_csv.py

# 3. Run modeling (generates plots + results.txt)
!python model_two_descriptor.py
```

### Outputs
- `two_descriptor_results.txt` - Performance metrics & feature importance
- `two_desc_parity.png` - Predicted vs Experimental scatter plot
- `two_desc_residuals.png` - Residual distribution
- `two_desc_importance.png` - Permutation importance of 2 descriptors

## 📈 Key Insights
1. **Both descriptors are equally important** (permutation R² drop ~0.44 each)
2. **PolarityProxy captures polarity penalty** - compounds with large water/octanol difference
3. **Linear relationship is optimal** - no benefit from non-linear models with current features
4. **Performance ceiling** - further improvement requires structural descriptors (TPSA, HBD/HBA)

## 🔬 Next Steps
- Add molecular descriptors if SMILES strings become available
- Test ensemble methods (though linear is likely optimal for 2 features)
- Investigate high-residual compounds for chemistry insights

## 📦 Dependencies
```
pandas
scikit-learn
tpot
matplotlib
seaborn
odfpy
```

## 📜 License
[Add your license here]
