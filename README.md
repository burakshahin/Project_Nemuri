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

**Core:**
- `logP_calculation.ods` - Source dataset (123 compounds)
- `read_ods_to_csv.py` - ODS→CSV converter
- `requirements.txt` - Python dependencies
- `tpot/` - Vendored TPOT library

**Modeling Scripts:**
- `model_two_descriptor.py` - Simple LinearRegression baseline (fast, 2 descriptors)
- `tpot_full_potential.py` - 🔥 **TPOT FULL POWER** - 100 gen × 100 pop, 3 descriptors + interactions

**Colab Notebooks:**
- `Colab_Notebook.ipynb` - Basic workflow
- `Colab_TPOT_Full_Potential.ipynb` - **RECOMMENDED** - Full AutoML with visualizations

## 🚀 Google Colab Workflow

### Quick Start (Simple LinearRegression)
```python
!git clone https://github.com/burakshahin/Project_Nemuri.git
%cd Project_Nemuri/Beta
!pip install pandas scikit-learn matplotlib seaborn odfpy
!python read_ods_to_csv.py
!python model_two_descriptor.py
```

### 🔥 TPOT Full Potential Mode (RECOMMENDED!)
```python
!git clone https://github.com/burakshahin/Project_Nemuri.git
%cd Project_Nemuri/Beta
!pip install pandas scikit-learn matplotlib seaborn odfpy tpot
!python read_ods_to_csv.py
!python tpot_full_potential.py  # 1-4 hours, 10,000 pipelines!
```

**OR** use the ready-made notebook: `Colab_TPOT_Full_Potential.ipynb`

### Outputs

**Simple Model:**
- `two_descriptor_results.txt` - Basic metrics
- `two_desc_*.png` - Basic plots

**TPOT Full Potential:**
- `tpot_full_results.txt` - Comprehensive analysis with confidence intervals
- `tpot_full_results.png` - 4-panel publication-ready figure
- `tpot_best_pipeline.py` - Exportable sklearn pipeline code
- Bootstrap CI, permutation importance (50 repeats), OOF predictions

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
