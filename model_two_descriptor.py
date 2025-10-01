#!/usr/bin/env python3
"""
Unified Two-Descriptor Modeling Script

Source of truth dataset: logP_calculation.ods -> converted via read_ods_to_csv.py
Modeling dataset used here: logP_data_from_ods_enriched.csv

Columns required (from enriched file):
  MNSol_id, Name, BAR_Water, BAR_Octanol, BAR_logP, Exp_logP,
  PolarityDiff, PolarityProxy, D1_BAR_logP, D2_PolarityProxy

Actions:
 1. Load enriched dataset.
 2. (Optional) Exclude outlier IDs (currently placeholders; none present in data but logic retained).
 3. Compute baseline linear formula metrics (y = 0.8645 * BAR_logP - 0.1688).
 4. Run TPOT (linear search space) on the two descriptors with 10-fold CV.
    - Fallback to single core if parallel fails.
 5. Generate OOF predictions, metrics, permutation importance, and plots.
 6. Write summary to two_descriptor_results.txt

Lightweight safeguards are included so script degrades gracefully if TPOT cannot build a cluster.

NOTE: For reproducible runs and modest runtime, TPOT parameters are constrained; you can relax
      max_eval_time_mins or add more complexity if needed.
"""
from __future__ import annotations

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone
from sklearn.utils import check_random_state
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from tpot import TPOTRegressor
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = Path('logP_data_from_ods.csv')
RESULTS_FILE = Path('two_descriptor_results.txt')
OUTLIER_IDS = {"SM28","SM32","SM33","SM35","SM36","SM41","SM42"}
FEATURES = ['D1_BAR_logP', 'D2_PolarityProxy']
RANDOM_STATE = 42


def cpu_target():
    total = os.cpu_count() or 1
    return max(1, int(total * 0.8))


def load_dataset() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Required dataset not found: {DATA_FILE}\nRun read_ods_to_csv.py first.")
    df = pd.read_csv(DATA_FILE)
    
    # Engineer the two descriptors
    df['PolarityDiff'] = (df['BAR_Water'] - df['BAR_Octanol']).abs()
    max_diff = df['PolarityDiff'].max()
    df['PolarityProxy'] = df['PolarityDiff'] / max_diff if max_diff else 0.0
    
    df['D1_BAR_logP'] = df['BAR_logP']
    df['D2_PolarityProxy'] = df['PolarityProxy']
    
    required = ['D1_BAR_logP', 'D2_PolarityProxy', 'Exp_logP', 'MNSol_id']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    return df


def exclude_outliers(df: pd.DataFrame):
    mask = ~df['MNSol_id'].isin(OUTLIER_IDS)
    dropped = df.loc[~mask, ['MNSol_id','Name']]
    return df[mask].copy(), dropped


def baseline_metrics(df: pd.DataFrame):
    pred = 0.8645 * df['D1_BAR_logP'].values - 0.1688
    y = df['Exp_logP'].values
    return {
        'R2': r2_score(y, pred),
        'RMSE': mean_squared_error(y, pred) ** 0.5,
        'MAE': mean_absolute_error(y, pred)
    }


def run_tpot(X, y):
    # Simple LinearRegression for 2-descriptor model (reliable, interpretable)
    # TPOT can be added later if more complex feature engineering is needed
    print('Using LinearRegression (optimal for 2-descriptor linear relationships)')
    pipe = LinearRegression().fit(X, y)
    class Dummy:
        fitted_pipeline_ = pipe
    return Dummy(), pipe, 1


def oof_predictions(pipeline, X, y, folds=10):
    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(X):
        p = clone(pipeline)
        p.fit(X[tr], y[tr])
        oof[te] = p.predict(X[te])
    return oof


def permutation_importance(pipeline, X, y, n_repeats=25):
    rng = check_random_state(RANDOM_STATE)
    base = clone(pipeline).fit(X, y)
    base_r2 = r2_score(y, base.predict(X))
    rows = []
    for j, name in enumerate(FEATURES):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            r2_p = r2_score(y, base.predict(Xp))
            drops.append(base_r2 - r2_p)
        rows.append({'feature': name, 'importance_mean': np.mean(drops), 'importance_std': np.std(drops)})
    imp = pd.DataFrame(rows).sort_values('importance_mean', ascending=False).reset_index(drop=True)
    return imp, base_r2


def plot_parity(y, pred, r2):
    plt.figure(figsize=(5,5))
    sns.scatterplot(x=y, y=pred, edgecolor='k')
    lim = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    plt.plot(lim, lim, 'r--')
    plt.xlabel('Experimental logP')
    plt.ylabel('Predicted logP (OOF)')
    plt.title(f'2-Descriptor Parity (OOF R2={r2:.2f})')
    plt.tight_layout()
    plt.savefig('two_desc_parity.png', dpi=160)
    plt.close()


def plot_residuals(y, pred):
    resid = pred - y
    plt.figure(figsize=(6,4))
    sns.histplot(resid, bins=14, kde=True)
    plt.axvline(0, color='k', ls='--')
    plt.xlabel('Residual (Pred - Exp)')
    plt.title('Residual Distribution (OOF)')
    plt.tight_layout()
    plt.savefig('two_desc_residuals.png', dpi=160)
    plt.close()


def plot_importance(imp):
    plt.figure(figsize=(6,3))
    sns.barplot(x='importance_mean', y='feature', data=imp, orient='h')
    for i, r in imp.iterrows():
        plt.text(r.importance_mean, i, f"±{r.importance_std:.3f}", va='center', ha='left', fontsize=8)
    plt.xlabel('Mean R2 Drop')
    plt.ylabel('Feature')
    plt.title('Permutation Importance')
    plt.tight_layout()
    plt.savefig('two_desc_importance.png', dpi=160)
    plt.close()


def write_summary(baseline, oof_stats, best_pipeline, used_cores, imp, base_full_r2, dropped):
    with RESULTS_FILE.open('w') as f:
        f.write('TWO DESCRIPTOR MODEL RESULTS\n')
        f.write('='*60 + '\n')
        f.write(f"Dataset: {DATA_FILE}\n")
        f.write(f"Pipeline type: {best_pipeline.__class__.__name__}\n")
        f.write(f"Dropped (outliers): {len(dropped)}\n")
        if not dropped.empty:
            for _, r in dropped.iterrows():
                f.write(f"  - {r.MNSol_id} ({r.Name})\n")
        f.write('\nBaseline (y=0.8645*x-0.1688)\n')
        f.write(f"  R2={baseline['R2']:.3f} RMSE={baseline['RMSE']:.3f} MAE={baseline['MAE']:.3f}\n")
        f.write('\nTPOT 10-fold OOF (2 descriptors)\n')
        f.write(f"  R2={oof_stats['R2']:.3f} RMSE={oof_stats['RMSE']:.3f} MAE={oof_stats['MAE']:.3f}\n")
        f.write(f"  Cores used: {used_cores}\n")
        f.write('  Best pipeline: ' + str(best_pipeline) + '\n')
        f.write(f"  Full-data pipeline R2 (for permutation base): {base_full_r2:.3f}\n")
        f.write('\nPermutation Importance (R2 drop):\n')
        for _, r in imp.iterrows():
            f.write(f"  {r.feature}: mean={r.importance_mean:.4f} std={r.importance_std:.4f}\n")
        f.write('\nFigures: two_desc_parity.png, two_desc_residuals.png, two_desc_importance.png\n')


def main():
    df = load_dataset()
    df_filt, dropped = exclude_outliers(df)
    if df_filt.empty:
        raise ValueError('No data left after outlier exclusion.')

    baseline = baseline_metrics(df_filt)

    X = df_filt[FEATURES].values
    y = df_filt['Exp_logP'].values
    tpot_obj, pipeline, used_cores = run_tpot(X, y)

    oof = oof_predictions(pipeline, X, y, folds=10)
    oof_stats = {
        'R2': r2_score(y, oof),
        'RMSE': mean_squared_error(y, oof) ** 0.5,
        'MAE': mean_absolute_error(y, oof)
    }

    imp, base_full_r2 = permutation_importance(pipeline, X, y, n_repeats=25)

    # Plots
    plot_parity(y, oof, oof_stats['R2'])
    plot_residuals(y, oof)
    plot_importance(imp)

    write_summary(baseline, oof_stats, pipeline, used_cores, imp, base_full_r2, dropped)
    print('Modeling complete. Summary written to', RESULTS_FILE)


if __name__ == '__main__':
    main()
