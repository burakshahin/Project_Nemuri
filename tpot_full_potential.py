#!/usr/bin/env python3
"""
TPOT Full Potential - 2-Descriptor logP Prediction with AutoML

This script leverages TPOT v2's evolutionary algorithm to find the optimal
pipeline for predicting experimental logP from BAR calculations.

Key Features:
- Extended search time (100 generations × 100 population)
- Multiple descriptor configurations
- Comprehensive validation (10-fold CV + Bootstrap)
- Permutation importance analysis
- Full reporting with confidence intervals

Designed for Google Colab with long runtime (~2-4 hours recommended).
Compatible with TPOT v2 API (vendored version in ./tpot/).
"""
from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns
from tpot import TPOTRegressor
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = Path('logP_data_from_ods.csv')
RESULTS_FILE = Path('tpot_full_results.txt')
OUTLIER_IDS = {"SM28","SM32","SM33","SM35","SM36","SM41","SM42"}
RANDOM_STATE = 42

# TPOT v2 API parameters
SEARCH_SPACE = 'linear-light'  # Options: 'linear', 'linear-light', 'graph', 'graph-light'
GENERATIONS = 100
POPULATION_SIZE = 100
CV_FOLDS = 10
MAX_TIME_MINS = 240  # 4 hours
EARLY_STOP = 20
N_BOOTSTRAP = 100
PERMUTATION_REPEATS = 50

print("="*70)
print("TPOT FULL POTENTIAL MODE - 2-DESCRIPTOR logP PREDICTION")
print("="*70)
print(f"Configuration:")
print(f"  Search Space: {SEARCH_SPACE}")
print(f"  Generations: {GENERATIONS}")
print(f"  Population: {POPULATION_SIZE}")
print(f"  CV Folds: {CV_FOLDS}")
print(f"  Max Time: {MAX_TIME_MINS} mins")
print(f"  Early stop: {EARLY_STOP} generations")
print(f"  Expected runtime: 1-4 hours (let it finish!)")
print("="*70)
print()


def load_and_prepare_data(use_interaction=True) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load data and engineer descriptors."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}\nRun read_ods_to_csv.py first!")
    
    df = pd.read_csv(DATA_FILE)
    
    # Engineer descriptors
    df['PolarityDiff'] = (df['BAR_Water'] - df['BAR_Octanol']).abs()
    max_diff = df['PolarityDiff'].max()
    df['PolarityProxy'] = df['PolarityDiff'] / max_diff if max_diff else 0.0
    
    df['D1_BAR_logP'] = df['BAR_logP']
    df['D2_PolarityProxy'] = df['PolarityProxy']
    
    # Optional interaction term
    if use_interaction:
        df['D3_Interaction'] = df['D1_BAR_logP'] * df['D2_PolarityProxy']
        feature_cols = ['D1_BAR_logP', 'D2_PolarityProxy', 'D3_Interaction']
    else:
        feature_cols = ['D1_BAR_logP', 'D2_PolarityProxy']
    
    # Filter outliers
    mask = ~df['MNSol_id'].isin(OUTLIER_IDS)
    df_clean = df[mask].copy()
    
    print(f"\nDataset loaded: {len(df_clean)} compounds (excluded {len(df)-len(df_clean)} outliers)")
    print(f"Descriptors: {', '.join(feature_cols)}")
    
    X = df_clean[feature_cols].values
    y = df_clean['Exp_logP'].values
    
    return df_clean, X, y


def baseline_performance(X, y):
    """Calculate baseline linear model for comparison."""
    print("\n" + "="*70)
    print("STEP 2: BASELINE PERFORMANCE")
    print("="*70)
    
    # Simple 2-descriptor model
    X_2d = X[:, :2]
    lr = LinearRegression()
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(lr, X_2d, y, cv=cv, scoring='r2')
    
    print(f"\nBASELINE (LinearRegression, 2 descriptors):")
    print(f"  R² = {scores.mean():.4f} ± {scores.std():.4f}")
    
    return scores.mean()


def run_tpot_search(X, y, test_mode=False):
    """
    Run TPOT evolutionary search using the vendored TPOT v2 API.
    
    test_mode: If True, use reduced parameters for quick testing.
    """
    print("\n" + "="*70)
    print("STEP 3: TPOT EVOLUTIONARY SEARCH")
    print("="*70)
    
    if test_mode:
        print("\nRunning TPOT with TEST configuration (fast mode)...")
        generations = 5
        population_size = 20
        cv = 3
        max_time = 10
        early_stop = 3
        print(f"  Generations: {generations}")
        print(f"  Population: {population_size}")
        print(f"  CV: {cv}-fold")
    else:
        print("\nStarting TPOT evolutionary search...")
        print("This will take 1-4 hours. Go grab coffee! ☕\n")
        generations = GENERATIONS
        population_size = POPULATION_SIZE
        cv = CV_FOLDS
        max_time = MAX_TIME_MINS
        early_stop = EARLY_STOP
    
    try:
        # TPOT v2 API: TPOTRegressor has its own signature
        # classification parameter is set automatically for regressors
        model = TPOTRegressor(
            search_space=SEARCH_SPACE,
            scorers=['r2'],  # Primary scorer
            scorers_weights=[1],  # Weight for r2
            cv=cv,
            max_time_mins=max_time,
            max_eval_time_mins=10,
            early_stop=early_stop,
            n_jobs=-1,
            verbose=2,
            random_state=RANDOM_STATE,
            # Pass population_size and generations via kwargs
            population_size=population_size,
            generations=generations
        )
        
        model.fit(X, y)
        print(f"\n✅ TPOT search complete!")
        
        # Try to export best pipeline if available
        try:
            if hasattr(model, 'fitted_pipeline_'):
                print(f"Best pipeline: {model.fitted_pipeline_}")
            if hasattr(model, 'export'):
                model.export('tpot_best_pipeline.py')
                print("📄 Best pipeline exported to: tpot_best_pipeline.py")
        except Exception as export_err:
            print(f"Note: Could not export pipeline: {export_err}")
        
        return model
        
    except Exception as e:
        print(f"\n❌ TPOT failed: {e}")
        print("Falling back to LinearRegression...")
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        return model


def out_of_fold_predictions(model, X, y):
    """Generate out-of-fold predictions for unbiased evaluation."""
    print("\n" + "="*70)
    print("STEP 4: OUT-OF-FOLD VALIDATION")
    print("="*70)
    
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.zeros_like(y)
    
    print(f"\nGenerating {CV_FOLDS}-fold OOF predictions...")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X), 1):
        model_fold = clone(model)
        model_fold.fit(X[train_idx], y[train_idx])
        y_pred[val_idx] = model_fold.predict(X[val_idx])
        print(f"  Fold {fold}/{CV_FOLDS} complete")
    
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    print(f"✅ OOF Results: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    
    return y_pred, r2, rmse, mae


def evaluate_with_bootstrap(model, X, y, n_bootstrap=100):
    """Bootstrap validation for confidence intervals."""
    print("\n" + "="*70)
    print("STEP 5: BOOTSTRAP CONFIDENCE INTERVALS")
    print("="*70)
    
    rng = np.random.RandomState(RANDOM_STATE)
    bootstrap_scores = []
    
    print(f"\nBootstrap validation ({n_bootstrap} samples)...")
    for i in range(n_bootstrap):
        if (i + 1) % 20 == 0:
            print(f"  Bootstrap {i+1}/{n_bootstrap}...")
        
        # Resample with replacement
        indices = rng.choice(len(X), size=len(X), replace=True)
        oob_indices = np.array([idx for idx in range(len(X)) if idx not in indices])
        
        if len(oob_indices) == 0:
            continue
        
        model_bs = clone(model)
        model_bs.fit(X[indices], y[indices])
        y_pred_oob = model_bs.predict(X[oob_indices])
        score = r2_score(y[oob_indices], y_pred_oob)
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    ci_lower = np.percentile(bootstrap_scores, 2.5)
    ci_upper = np.percentile(bootstrap_scores, 97.5)
    
    print(f"✅ Bootstrap complete:")
    print(f"  R² = {bootstrap_scores.mean():.4f} ± {bootstrap_scores.std():.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return bootstrap_scores


def permutation_importance(model, X, y, n_repeats=50):
    """Calculate permutation importance with confidence intervals."""
    print("\n" + "="*70)
    print("STEP 6: PERMUTATION IMPORTANCE")
    print("="*70)
    
    print(f"\nComputing permutation importance ({n_repeats} repeats)...")
    
    # Baseline score
    y_pred = model.predict(X)
    baseline_score = r2_score(y, y_pred)
    
    feature_names = ['D1_BAR_logP', 'D2_PolarityProxy']
    if X.shape[1] == 3:
        feature_names.append('D3_Interaction')
    
    importances = {}
    rng = np.random.RandomState(RANDOM_STATE)
    
    for feature_idx, feature_name in enumerate(feature_names):
        print(f"  Testing {feature_name}...")
        scores = []
        
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, feature_idx] = rng.permutation(X_permuted[:, feature_idx])
            y_pred_perm = model.predict(X_permuted)
            score_perm = r2_score(y, y_pred_perm)
            importance = baseline_score - score_perm
            scores.append(importance)
        
        importances[feature_name] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'ci_lower': np.percentile(scores, 2.5),
            'ci_upper': np.percentile(scores, 97.5)
        }
    
    print("✅ Permutation importance complete")
    return importances


def plot_results(y_true, y_pred, importances, baseline_r2, final_r2):
    """Generate comprehensive visualization."""
    print("\n" + "="*70)
    print("STEP 7: VISUALIZATION")
    print("="*70)
    
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Parity plot
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
            'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Experimental logP', fontsize=12)
    ax.set_ylabel('Predicted logP', fontsize=12)
    ax.set_title(f'Parity Plot (R² = {final_r2:.4f})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Residuals
    ax = axes[0, 1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.axhline(0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted logP', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 3. Residual histogram
    ax = axes[1, 0]
    ax.hist(residuals, bins=20, edgecolor='k', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Residuals', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 4. Feature importance
    ax = axes[1, 1]
    feature_names = list(importances.keys())
    means = [importances[f]['mean'] for f in feature_names]
    stds = [importances[f]['std'] for f in feature_names]
    
    colors = ['#2ecc71' if m > 0 else '#e74c3c' for m in means]
    bars = ax.barh(feature_names, means, xerr=stds, color=colors, 
                   alpha=0.7, edgecolor='k', linewidth=1.5)
    ax.axvline(0, color='k', linestyle='-', lw=1)
    ax.set_xlabel('Importance (Δ R²)', fontsize=12)
    ax.set_title('Permutation Importance', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('tpot_full_results.png', dpi=300, bbox_inches='tight')
    print("📊 Plot saved: tpot_full_results.png")


def save_comprehensive_report(baseline_r2, final_r2, rmse, mae, 
                             bootstrap_scores, importances):
    """Save detailed text report."""
    print("\n" + "="*70)
    print("STEP 8: COMPREHENSIVE REPORT")
    print("="*70)
    
    with open(RESULTS_FILE, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TPOT FULL POTENTIAL - COMPREHENSIVE RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write(f"  Search Space: {SEARCH_SPACE}\n")
        f.write(f"  Generations: {GENERATIONS}\n")
        f.write(f"  Population: {POPULATION_SIZE}\n")
        f.write(f"  CV Folds: {CV_FOLDS}\n")
        f.write(f"  Max Time: {MAX_TIME_MINS} mins\n")
        f.write(f"  Early Stop: {EARLY_STOP} generations\n\n")
        
        f.write("PERFORMANCE COMPARISON:\n")
        f.write(f"  Baseline (LinearRegression): R² = {baseline_r2:.4f}\n")
        f.write(f"  TPOT Best Model: R² = {final_r2:.4f}\n")
        improvement = ((final_r2 - baseline_r2) / baseline_r2) * 100
        f.write(f"  Improvement: {improvement:+.2f}%\n\n")
        
        f.write("OUT-OF-FOLD METRICS:\n")
        f.write(f"  R² = {final_r2:.4f}\n")
        f.write(f"  RMSE = {rmse:.4f}\n")
        f.write(f"  MAE = {mae:.4f}\n\n")
        
        f.write(f"BOOTSTRAP CONFIDENCE (n={len(bootstrap_scores)}):\n")
        f.write(f"  Mean R² = {bootstrap_scores.mean():.4f} ± {bootstrap_scores.std():.4f}\n")
        f.write(f"  95% CI = [{np.percentile(bootstrap_scores, 2.5):.4f}, ")
        f.write(f"{np.percentile(bootstrap_scores, 97.5):.4f}]\n\n")
        
        f.write("PERMUTATION IMPORTANCE:\n")
        for feature, stats in importances.items():
            f.write(f"  {feature}:\n")
            f.write(f"    Mean = {stats['mean']:.4f} ± {stats['std']:.4f}\n")
            f.write(f"    95% CI = [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("="*70 + "\n")
        f.write("- R²: Fraction of variance explained (closer to 1.0 is better)\n")
        f.write("- RMSE: Root mean squared error in logP units\n")
        f.write("- MAE: Mean absolute error in logP units\n")
        f.write("- Permutation importance: Drop in R² when feature is shuffled\n")
        f.write("  (higher = more important for predictions)\n")
    
    print(f"📄 Comprehensive report saved: {RESULTS_FILE}")


def main(test_mode=False):
    """Main execution pipeline."""
    # Step 1: Load data
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    df, X, y = load_and_prepare_data(use_interaction=True)
    
    # Step 2: Baseline
    baseline_r2 = baseline_performance(X, y)
    
    # Step 3: TPOT search
    model = run_tpot_search(X, y, test_mode=test_mode)
    
    # Step 4: Out-of-fold validation
    y_pred, final_r2, rmse, mae = out_of_fold_predictions(model, X, y)
    
    # Step 5: Bootstrap
    n_bootstrap = 10 if test_mode else N_BOOTSTRAP
    bootstrap_scores = evaluate_with_bootstrap(model, X, y, n_bootstrap=n_bootstrap)
    
    # Step 6: Permutation importance
    n_repeats = 5 if test_mode else PERMUTATION_REPEATS
    importances = permutation_importance(model, X, y, n_repeats=n_repeats)
    
    # Step 7: Visualization
    plot_results(y, y_pred, importances, baseline_r2, final_r2)
    
    # Step 8: Report
    save_comprehensive_report(baseline_r2, final_r2, rmse, mae,
                            bootstrap_scores, importances)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"Results saved to: {RESULTS_FILE}")
    print("Plots saved to: tpot_full_results.png")
    if Path('tpot_best_pipeline.py').exists():
        print("Best pipeline: tpot_best_pipeline.py")
    print()


if __name__ == '__main__':
    # Check for test mode
    test_mode = '--test' in sys.argv
    
    if test_mode:
        print("\n⚠️  TEST MODE ENABLED (reduced parameters)\n")
    
    main(test_mode=test_mode)
