#!/usr/bin/env python3
"""
Two-Descriptor TPOT Analysis (Updated Dataset with New SM25-SM46 Rows)

User requirements implemented:
 - Use only 2 meaningful descriptors (Baseline BAR_logP and PolarityProxy = |BAR_Water - BAR_Octanol| normalized)
 - Remove aromatic descriptor entirely.
 - Exclude outlier IDs: SM28, SM32, SM33, SM35, SM36, SM41, SM42 (if present) from BOTH training and validation (i.e. they are dropped before CV / TPOT).
 - Apply user baseline linear relation y = 0.8645 * x - 0.1688 (x = BAR_logP) and report its performance on remaining data.
 - Run TPOT with "full potential" (broad linear search_space) and no overall time cap. (NOTE: In this environment earlier parallel attempts caused Dask worker errors; we implement an n_jobs=desired with fallback.)
 - Target CPU usage ~80%. Detected cores -> use floor(cores*0.8).

If the newly added SM25-SM46 compounds or the specified outliers aren't found, the script will warn but continue.

Outputs appended to: three_descriptor_results.txt
Figures produced:
 - two_desc_tpot_parity.png
 - two_desc_tpot_residuals.png
 - two_desc_tpot_permutation.png

Permutation importance (R2 drop) is computed on full-data fit for the two descriptors.

DISCLAIMER: TPOT 1.1.0 in this environment may still revert to single-thread if Dask cluster cannot spawn workers. Fallback logic is included.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import seaborn as sns
from tpot import TPOTRegressor
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
RESULTS_FILE = 'three_descriptor_results.txt'
OUTLIER_IDS = {"SM28","SM32","SM33","SM35","SM36","SM41","SM42"}
FEATURE_COLS = ['D1_BAR_logP','D2_PolarityProxy']


def cpu_target_cores():
    total = os.cpu_count() or 1
    return max(1, int(total * 0.8))


def load_and_engineer():
    df = pd.read_csv('logP_data.csv', skiprows=1)
    df.columns = ['MNSol_id','Name','BAR_Water','BAR_Octanol','BAR_logP',
                  'Exp_Water','Exp_Octanol','Exp_logP','RT','Unknown']
    df = df.dropna(subset=['BAR_logP','Exp_logP']).copy()
    diff = (df['BAR_Water'] - df['BAR_Octanol']).abs()
    df['D1_BAR_logP'] = df['BAR_logP']
    mx = diff.max() if diff.max() else 1.0
    df['D2_PolarityProxy'] = diff / mx
    return df


def exclude_outliers(df):
    # Outliers referenced as IDs like SM28 etc. Could appear in MNSol_id or Name.
    mask = ~df['MNSol_id'].isin(OUTLIER_IDS) & ~df['Name'].isin(OUTLIER_IDS)
    dropped = df.loc[~mask, ['MNSol_id','Name']]
    return df[mask].copy(), dropped


def user_baseline_formula(df):
    # y = 0.8645 * BAR_logP - 0.1688
    pred = 0.8645 * df['D1_BAR_logP'].values - 0.1688
    y = df['Exp_logP'].values
    r2 = r2_score(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae = mean_absolute_error(y, pred)
    return {'r2': r2, 'rmse': rmse, 'mae': mae}


def run_tpot(df):
    X = df[FEATURE_COLS].values
    y = df['Exp_logP'].values
    desired_cores = cpu_target_cores()
    # Attempt with parallel first; fallback to single-core on failure.
    for cores in [desired_cores, 1]:
        try:
            tpot = TPOTRegressor(
                search_space='linear',
                scorers=['r2'],
                scorers_weights=[1],
                cv=10,
                max_time_mins=float('inf'),  # no global cap
                max_eval_time_mins=5,
                n_jobs=cores,
                verbose=2,
                random_state=RANDOM_STATE,
                allow_inner_regressors=True
            )
            tpot.fit(X, y)
            best = tpot.fitted_pipeline_
            return tpot, best, cores
        except Exception as e:
            print(f"TPOT run failed with n_jobs={cores}: {e}. Trying fallback...")
            continue
    raise RuntimeError("TPOT failed in both parallel and single-core modes.")


def oof_predictions(pipeline, X, y, folds=10):
    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros_like(y)
    for tr, te in kf.split(X):
        p = clone(pipeline)
        p.fit(X[tr], y[tr])
        oof[te] = p.predict(X[te])
    return oof


def permutation_importance_two(pipeline, X, y, n_repeats=25):
    rng = check_random_state(RANDOM_STATE)
    model = clone(pipeline)
    model.fit(X, y)
    base_r2 = r2_score(y, model.predict(X))
    imps = []
    for j, fname in enumerate(FEATURE_COLS):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            r2_p = r2_score(y, model.predict(Xp))
            drops.append(base_r2 - r2_p)
        imps.append({'feature': fname, 'importance_mean': np.mean(drops), 'importance_std': np.std(drops)})
    imp_df = pd.DataFrame(imps).sort_values('importance_mean', ascending=False).reset_index(drop=True)
    return imp_df, base_r2


def plot_parity(y, pred, title, fname):
    plt.figure(figsize=(5,5))
    sns.scatterplot(x=y, y=pred, edgecolor='k')
    lims = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel('Experimental logP')
    plt.ylabel('Predicted logP (OOF)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()


def plot_residuals(y, pred, title, fname):
    resid = pred - y
    plt.figure(figsize=(6,4))
    sns.histplot(resid, bins=12, kde=True)
    plt.axvline(0, color='k', ls='--')
    plt.xlabel('Residual (Pred - Exp)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()


def plot_importance(imp_df, fname):
    plt.figure(figsize=(6,3))
    sns.barplot(x='importance_mean', y='feature', data=imp_df, orient='h', palette='viridis')
    plt.xlabel('Mean R2 Drop')
    plt.ylabel('Feature')
    plt.title('Permutation Importance (2 Descriptors)')
    for i, row in imp_df.iterrows():
        plt.text(row.importance_mean, i, f" ±{row.importance_std:.3f}", va='center', ha='left', fontsize=8)
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()


def append_results(baseline_res, oof_r2, oof_rmse, oof_mae, best_pipeline, used_cores, dropped_df, imp_df, base_full_r2):
    with open(RESULTS_FILE, 'a') as f:
        f.write('\n' + '='*70 + '\n')
        f.write('TWO DESCRIPTOR TPOT (Updated Dataset, Outliers Excluded)\n')
        f.write('='*70 + '\n')
        f.write(f"Outlier IDs requested to drop: {sorted(list(OUTLIER_IDS))}\n")
        if not dropped_df.empty:
            f.write('Dropped rows found in dataset:\n')
            for _, r in dropped_df.iterrows():
                f.write(f"  - {r.MNSol_id} ({r.Name})\n")
        else:
            f.write('Warning: None of the specified outlier IDs were present in this dataset snapshot.\n')
        f.write('\nUser Baseline Linear (y = 0.8645 * BAR_logP - 0.1688):\n')
        f.write(f"  R2={baseline_res['r2']:.3f} RMSE={baseline_res['rmse']:.3f} MAE={baseline_res['mae']:.3f}\n")
        f.write('\nTPOT (10-fold OOF) with 2 descriptors:\n')
        f.write(f"  R2={oof_r2:.3f} RMSE={oof_rmse:.3f} MAE={oof_mae:.3f}\n")
        f.write(f"  CPU cores target used: {used_cores}\n")
        f.write('  Best pipeline steps: ' + str(best_pipeline) + '\n')
        f.write(f"  Full-data baseline R2 for permutation ref: {base_full_r2:.3f}\n")
        f.write('\nPermutation Importance (R2 drop):\n')
        for _, row in imp_df.iterrows():
            f.write(f"  {row.feature}: mean={row.importance_mean:.4f} std={row.importance_std:.4f}\n")
        f.write('Figures: two_desc_tpot_parity.png, two_desc_tpot_residuals.png, two_desc_tpot_permutation.png\n')


def main():
    df = load_and_engineer()
    df_filtered, dropped = exclude_outliers(df)
    if df_filtered.empty:
        raise ValueError('All rows were filtered out. Check outlier list or dataset.')

    baseline_res = user_baseline_formula(df_filtered)

    tpot_obj, best_pipeline, used_cores = run_tpot(df_filtered)

    X = df_filtered[FEATURE_COLS].values
    y = df_filtered['Exp_logP'].values
    oof = oof_predictions(best_pipeline, X, y, folds=10)
    oof_r2 = r2_score(y, oof)
    oof_rmse = np.sqrt(mean_squared_error(y, oof))
    oof_mae = mean_absolute_error(y, oof)

    # Plots
    plot_parity(y, oof, f'TPOT 2-Descriptor OOF Parity (R2={oof_r2:.2f})', 'two_desc_tpot_parity.png')
    plot_residuals(y, oof, 'TPOT 2-Descriptor OOF Residuals', 'two_desc_tpot_residuals.png')

    # Permutation importance
    imp_df, base_full_r2 = permutation_importance_two(best_pipeline, X, y, n_repeats=25)
    plot_importance(imp_df, 'two_desc_tpot_permutation.png')

    append_results(baseline_res, oof_r2, oof_rmse, oof_mae, best_pipeline, used_cores, dropped, imp_df, base_full_r2)
    print('Two-descriptor TPOT analysis complete.')


if __name__ == '__main__':
    main()
