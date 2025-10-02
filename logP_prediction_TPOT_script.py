"""
LogP Prediction using TPOT AutoML - Complete Script
====================================================

This script uses TPOT (Tree-based Pipeline Optimization Tool) to automatically 
find the best machine learning pipeline for predicting experimental logP from 
calculated values.

Goals:
- Improve upon 80% linear regression accuracy
- Remove 7 identified outliers (SM28, SM32, SM33, SM35, SM36, SM41, SM42)
- Engineer meaningful molecular descriptors
- Apply proper cross-validation to prevent overfitting
- Use TPOT to find optimal preprocessing + model pipeline

Configuration:
- Maximum TPOT settings: 150 generations, 100 population, 480 minutes (8 hours)
- Expected to achieve 90%+ R² accuracy

Usage:
    python logP_prediction_TPOT_script.py

Requirements:
    pip install tpot scikit-learn pandas numpy matplotlib seaborn joblib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
import warnings
import joblib
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("LogP Prediction with TPOT AutoML")
print("="*80)
print("\n✅ Libraries imported successfully!\n")

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("="*80)
print("STEP 1: Loading Data")
print("="*80)

# Load the data
df = pd.read_csv('logP_data_from_ods.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Check for missing values and data info
print("\n\nDataset Info:")
print(df.info())
print("\n\nMissing values per column:")
print(df.isnull().sum())

# ============================================================================
# 2. REMOVE OUTLIERS AND CLEAN DATA
# ============================================================================
print("\n\n" + "="*80)
print("STEP 2: Removing Outliers")
print("="*80)

# Define outliers to remove
outliers = ['SM28', 'SM32', 'SM33', 'SM35', 'SM36', 'SM41', 'SM42']

print(f"\nOriginal dataset size: {len(df)}")
print(f"Outliers to remove: {outliers}")

# Remove outliers
df_clean = df[~df['MNSol_id'].isin(outliers)].copy()

print(f"Dataset size after removing outliers: {len(df_clean)}")
print(f"Removed {len(df) - len(df_clean)} samples")

# Remove rows with missing Exp_logP values (our target variable)
df_clean = df_clean.dropna(subset=['Exp_logP'])

print(f"Dataset size after removing missing targets: {len(df_clean)}")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n\n" + "="*80)
print("STEP 3: Feature Engineering")
print("="*80)

def engineer_features(df):
    """
    Engineer meaningful molecular descriptors from BAR calculations
    """
    df_feat = df.copy()
    
    # Original features
    df_feat['BAR_logP'] = df['BAR_logP']
    df_feat['BAR_Water'] = df['BAR_Water']
    df_feat['BAR_Octanol'] = df['BAR_Octanol']
    
    # 1. Energy-based descriptors
    # Total solvation energy
    df_feat['Total_Solvation_Energy'] = df['BAR_Water'] + df['BAR_Octanol']
    
    # Absolute energy difference (partition strength)
    df_feat['Energy_Difference_Abs'] = np.abs(df['BAR_Water'] - df['BAR_Octanol'])
    
    # Solvation ratio (avoids division by zero)
    df_feat['Solvation_Ratio'] = df['BAR_Water'] / (df['BAR_Octanol'] + 1e-6)
    
    # 2. Non-linear transformations
    # These can capture non-linear relationships
    df_feat['BAR_logP_squared'] = df['BAR_logP'] ** 2
    df_feat['BAR_logP_cubed'] = df['BAR_logP'] ** 3
    df_feat['BAR_logP_sqrt'] = np.sqrt(np.abs(df['BAR_logP']))  * np.sign(df['BAR_logP'])
    
    # Log transformations (shifted to avoid log(0))
    df_feat['log_BAR_Water'] = np.log(np.abs(df['BAR_Water']) + 1) * np.sign(df['BAR_Water'])
    df_feat['log_BAR_Octanol'] = np.log(np.abs(df['BAR_Octanol']) + 1) * np.sign(df['BAR_Octanol'])
    
    # 3. Partition behavior indicators
    # Hydrophilicity indicator
    df_feat['Hydrophilic_Score'] = df['BAR_Water'] / (np.abs(df['BAR_Water']) + np.abs(df['BAR_Octanol']) + 1e-6)
    
    # Lipophilicity indicator
    df_feat['Lipophilic_Score'] = np.abs(df['BAR_Octanol']) / (np.abs(df['BAR_Water']) + np.abs(df['BAR_Octanol']) + 1e-6)
    
    # 4. Interaction terms
    df_feat['Water_Octanol_Product'] = df['BAR_Water'] * df['BAR_Octanol']
    
    # 5. Linear correction from Excel formula: y = 0.8645x - 0.1688
    df_feat['Linear_Corrected_logP'] = 0.8645 * df['BAR_logP'] - 0.1688
    
    # 6. Deviation from linear prediction
    df_feat['Linear_Deviation'] = df['BAR_logP'] - df_feat['Linear_Corrected_logP']
    
    # 7. Energy normalized by logP
    df_feat['Energy_per_logP'] = df_feat['Total_Solvation_Energy'] / (np.abs(df['BAR_logP']) + 1e-6)
    
    # 8. Categorize molecules by water solubility
    df_feat['High_Water_Solubility'] = (df['BAR_Water'] > df['BAR_Water'].median()).astype(int)
    df_feat['High_Octanol_Solubility'] = (df['BAR_Octanol'] < df['BAR_Octanol'].median()).astype(int)
    
    return df_feat

# Apply feature engineering
df_features = engineer_features(df_clean)

print(f"\nNumber of engineered features: {len(df_features.columns)}")
print(f"\nNew features created:")
new_features = [col for col in df_features.columns if col not in df_clean.columns]
for i, feat in enumerate(new_features, 1):
    print(f"{i}. {feat}")

# ============================================================================
# 4. PREPARE FEATURES AND TARGET
# ============================================================================
print("\n\n" + "="*80)
print("STEP 4: Preparing Features and Target")
print("="*80)

# Select features for modeling
feature_columns = [col for col in df_features.columns 
                   if col not in ['MNSol_id', 'Name', 'Exp_Water', 'Exp_Octanol', 'Exp_logP']]

X = df_features[feature_columns].values
y = df_features['Exp_logP'].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"\nFeatures used for modeling:")
for i, feat in enumerate(feature_columns, 1):
    print(f"{i}. {feat}")

# Check for any remaining NaN or infinite values
print("\n\nChecking for invalid values...")
print(f"NaN values in X: {np.isnan(X).sum()}")
print(f"Infinite values in X: {np.isinf(X).sum()}")
print(f"NaN values in y: {np.isnan(y).sum()}")

# Replace any inf values with large finite numbers
X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

print("\n✅ Data cleaning complete!")

# ============================================================================
# 5. SPLIT DATA FOR TRAINING AND TESTING
# ============================================================================
print("\n\n" + "="*80)
print("STEP 5: Splitting Data")
print("="*80)

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Test set proportion: {X_test.shape[0] / len(X) * 100:.1f}%")

# ============================================================================
# 6. BASELINE: LINEAR REGRESSION BENCHMARK
# ============================================================================
print("\n\n" + "="*80)
print("STEP 6: Baseline Model (Ridge Regression)")
print("="*80)

# Create baseline pipeline with scaling + ridge regression
baseline_model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=1.0))
])

baseline_model.fit(X_train, y_train)

# Predictions
y_pred_baseline = baseline_model.predict(X_test)

# Evaluate
r2_baseline = r2_score(y_test, y_pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)

print("\n" + "="*60)
print("BASELINE MODEL PERFORMANCE (Ridge Regression)")
print("="*60)
print(f"R² Score: {r2_baseline:.4f} ({r2_baseline*100:.2f}% accuracy)")
print(f"RMSE: {rmse_baseline:.4f}")
print(f"MAE: {mae_baseline:.4f}")
print("="*60)

# Cross-validation score
cv_scores = cross_val_score(baseline_model, X_train, y_train, cv=5, 
                            scoring='r2', n_jobs=-1)
print(f"\nCross-validation R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# 7. TPOT: AUTOMATED MACHINE LEARNING
# ============================================================================
print("\n\n" + "="*80)
print("STEP 7: TPOT Automated Machine Learning")
print("="*80)

from tpot import TPOTRegressor

# Configure TPOT for maximum thoroughness
tpot_config = {
    'generations': 150,  # Maximum generations for exhaustive exploration
    'population_size': 100,  # Large population for maximum diversity
    'cv': 5,  # 5-fold cross-validation
    'random_state': 42,
    'verbosity': 2,  # Show progress
    'scoring': 'r2',  # Optimize for R² score
    'n_jobs': -1,  # Use all available cores
    'max_time_mins': 480,  # 8 hours - maximum thoroughness
    'max_eval_time_mins': 10,  # Allow more time for complex pipelines
    'early_stop': 20,  # More patience before stopping
}

print("\nTPOT Configuration:")
print("="*60)
for key, value in tpot_config.items():
    print(f"{key}: {value}")
print("="*60)
print("\n🚀 MAXIMUM THOROUGHNESS MODE ACTIVATED!")
print("Expected runtime: ~6-8 hours (480 minutes max)")
print("This will evaluate 15,000+ pipeline configurations")
print("With 5-fold CV: ~75,000 total model fits")
print("\n⚠️  IMPORTANT: This will take a long time!")
print("The run may finish earlier if early stopping triggers")
print("You can interrupt anytime and still get the best pipeline found so far\n")

# Initialize TPOT
tpot = TPOTRegressor(**tpot_config)

print("Starting TPOT optimization...")
print("Please be patient - this may take several hours!\n")

# Fit TPOT
tpot.fit(X_train, y_train)

print("\n" + "="*60)
print("TPOT OPTIMIZATION COMPLETE!")
print("="*60)

# ============================================================================
# 8. EVALUATE TPOT MODEL
# ============================================================================
print("\n\n" + "="*80)
print("STEP 8: Evaluating TPOT Model")
print("="*80)

# Make predictions with TPOT model
y_pred_tpot = tpot.predict(X_test)

# Calculate metrics
r2_tpot = r2_score(y_test, y_pred_tpot)
rmse_tpot = np.sqrt(mean_squared_error(y_test, y_pred_tpot))
mae_tpot = mean_absolute_error(y_test, y_pred_tpot)

print("\n" + "="*60)
print("TPOT MODEL PERFORMANCE")
print("="*60)
print(f"R² Score: {r2_tpot:.4f} ({r2_tpot*100:.2f}% accuracy)")
print(f"RMSE: {rmse_tpot:.4f}")
print(f"MAE: {mae_tpot:.4f}")
print("="*60)

# Compare with baseline
print("\n" + "="*60)
print("COMPARISON: TPOT vs BASELINE")
print("="*60)
print(f"R² Improvement: {(r2_tpot - r2_baseline):.4f} ({(r2_tpot - r2_baseline)*100:.2f}%)")
print(f"RMSE Improvement: {(rmse_baseline - rmse_tpot):.4f}")
print(f"MAE Improvement: {(mae_baseline - mae_tpot):.4f}")
print("="*60)

if r2_tpot > r2_baseline:
    print("\n✅ TPOT model outperforms baseline!")
else:
    print("\n⚠️  Baseline still competitive. Consider longer TPOT runtime.")

# ============================================================================
# 9. CROSS-VALIDATION ANALYSIS
# ============================================================================
print("\n\n" + "="*80)
print("STEP 9: Cross-Validation Analysis (Overfitting Check)")
print("="*80)

cv_results = cross_validate(
    tpot.fitted_pipeline_,
    X_train, 
    y_train,
    cv=5,
    scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
    return_train_score=True,
    n_jobs=-1
)

print("\n" + "="*60)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("="*60)

# R² scores
train_r2 = cv_results['train_r2']
test_r2 = cv_results['test_r2']

print(f"\nR² Score:")
print(f"  Training:   {train_r2.mean():.4f} (+/- {train_r2.std() * 2:.4f})")
print(f"  Validation: {test_r2.mean():.4f} (+/- {test_r2.std() * 2:.4f})")

# RMSE
train_rmse = np.sqrt(-cv_results['train_neg_mean_squared_error'])
test_rmse = np.sqrt(-cv_results['test_neg_mean_squared_error'])

print(f"\nRMSE:")
print(f"  Training:   {train_rmse.mean():.4f} (+/- {train_rmse.std() * 2:.4f})")
print(f"  Validation: {test_rmse.mean():.4f} (+/- {test_rmse.std() * 2:.4f})")

# MAE
train_mae = -cv_results['train_neg_mean_absolute_error']
test_mae = -cv_results['test_neg_mean_absolute_error']

print(f"\nMAE:")
print(f"  Training:   {train_mae.mean():.4f} (+/- {train_mae.std() * 2:.4f})")
print(f"  Validation: {test_mae.mean():.4f} (+/- {test_mae.std() * 2:.4f})")

# Overfitting check
overfit_indicator = train_r2.mean() - test_r2.mean()
print(f"\n{'='*60}")
print(f"Overfitting Indicator: {overfit_indicator:.4f}")
if overfit_indicator < 0.1:
    print("✅ Model generalizes well! Low overfitting.")
elif overfit_indicator < 0.2:
    print("⚠️  Moderate overfitting detected.")
else:
    print("❌ High overfitting! Model may not generalize well.")
print("="*60)

# ============================================================================
# 10. VISUALIZATIONS
# ============================================================================
print("\n\n" + "="*80)
print("STEP 10: Creating Visualizations")
print("="*80)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Predicted vs Actual (TPOT)
ax1 = axes[0, 0]
ax1.scatter(y_test, y_pred_tpot, alpha=0.6, s=100, edgecolors='k', linewidth=0.5)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Experimental logP', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted logP (TPOT)', fontsize=12, fontweight='bold')
ax1.set_title(f'TPOT Model: R² = {r2_tpot:.4f}', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. Predicted vs Actual (Baseline)
ax2 = axes[0, 1]
ax2.scatter(y_test, y_pred_baseline, alpha=0.6, s=100, 
            edgecolors='k', linewidth=0.5, color='orange')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Experimental logP', fontsize=12, fontweight='bold')
ax2.set_ylabel('Predicted logP (Baseline)', fontsize=12, fontweight='bold')
ax2.set_title(f'Baseline Model: R² = {r2_baseline:.4f}', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. Residuals Plot (TPOT)
ax3 = axes[1, 0]
residuals_tpot = y_test - y_pred_tpot
ax3.scatter(y_pred_tpot, residuals_tpot, alpha=0.6, s=100, edgecolors='k', linewidth=0.5)
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Predicted logP', fontsize=12, fontweight='bold')
ax3.set_ylabel('Residuals', fontsize=12, fontweight='bold')
ax3.set_title('TPOT Residuals Plot', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Model Comparison
ax4 = axes[1, 1]
models = ['Baseline\n(Ridge)', 'TPOT\n(Optimized)']
r2_scores = [r2_baseline, r2_tpot]
colors = ['orange', 'steelblue']

bars = ax4.bar(models, r2_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax4.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax4.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax4.set_ylim([0, 1])
ax4.axhline(y=0.8, color='green', linestyle='--', lw=2, label='80% Target')
ax4.legend(fontsize=10)

# Add value labels on bars
for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{score:.4f}\n({score*100:.2f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('logP_prediction_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved as 'logP_prediction_results.png'")

# ============================================================================
# 11. FEATURE IMPORTANCE (if available)
# ============================================================================
print("\n\n" + "="*80)
print("STEP 11: Feature Importance Analysis")
print("="*80)

try:
    # Get the final estimator from the pipeline
    final_estimator = None
    
    if hasattr(tpot.fitted_pipeline_, 'named_steps'):
        steps = list(tpot.fitted_pipeline_.named_steps.values())
        final_estimator = steps[-1]
    elif hasattr(tpot.fitted_pipeline_, 'steps'):
        final_estimator = tpot.fitted_pipeline_.steps[-1][1]
    
    # Extract feature importances
    if final_estimator is not None and hasattr(final_estimator, 'feature_importances_'):
        importances = final_estimator.feature_importances_
        
        # Create dataframe
        feature_importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE RANKING")
        print("="*60)
        print(feature_importance_df.to_string(index=False))
        print("="*60)
        
        # Plot top 15 features
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['Importance'], 
                color='steelblue', edgecolor='black', linewidth=1)
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance', fontsize=12, fontweight='bold')
        plt.title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n✅ Feature importance plot saved as 'feature_importance.png'")
        
    else:
        print("\n⚠️  Selected model doesn't support feature importance extraction.")
        
except Exception as e:
    print(f"\n⚠️  Could not extract feature importances: {str(e)}")

# ============================================================================
# 12. EXPORT PIPELINE AND MODEL
# ============================================================================
print("\n\n" + "="*80)
print("STEP 12: Exporting Pipeline and Model")
print("="*80)

# Export the best pipeline as Python code
tpot.export('tpot_logP_pipeline.py')
print("\n✅ Best pipeline exported to 'tpot_logP_pipeline.py'")

# Save the entire TPOT fitted pipeline
joblib.dump(tpot.fitted_pipeline_, 'tpot_logP_model.pkl')
print("✅ Model saved as 'tpot_logP_model.pkl'")

# Display the pipeline
print("\n" + "="*60)
print("BEST PIPELINE FOUND BY TPOT:")
print("="*60)
print(tpot.fitted_pipeline_)
print("="*60)

# ============================================================================
# 13. PREDICTIONS ON ALL DATA
# ============================================================================
print("\n\n" + "="*80)
print("STEP 13: Making Predictions on All Data")
print("="*80)

# Create predictions for all data
X_all = df_features[feature_columns].values
X_all = np.nan_to_num(X_all, nan=0.0, posinf=1e6, neginf=-1e6)
y_all_pred = tpot.fitted_pipeline_.predict(X_all)

# Add predictions to dataframe
df_results = df_features[['MNSol_id', 'Name', 'BAR_logP', 'Exp_logP']].copy()
df_results['Predicted_logP'] = y_all_pred
df_results['Prediction_Error'] = df_results['Exp_logP'] - df_results['Predicted_logP']
df_results['Absolute_Error'] = np.abs(df_results['Prediction_Error'])

# Sort by absolute error
df_results_sorted = df_results.sort_values('Absolute_Error')

print("\n" + "="*80)
print("BEST PREDICTIONS (Top 10 with lowest error)")
print("="*80)
print(df_results_sorted.head(10).to_string(index=False))

print("\n" + "="*80)
print("WORST PREDICTIONS (Top 10 with highest error)")
print("="*80)
print(df_results_sorted.tail(10).to_string(index=False))

# Save results
df_results_sorted.to_csv('logP_predictions_detailed.csv', index=False)
print("\n✅ Detailed predictions saved to 'logP_predictions_detailed.csv'")

# ============================================================================
# 14. FINAL SUMMARY
# ============================================================================
print("\n\n" + "="*80)
print("FINAL SUMMARY AND RECOMMENDATIONS")
print("="*80)

print(f"\n📊 PERFORMANCE METRICS:")
print(f"   Baseline R²: {r2_baseline:.4f} ({r2_baseline*100:.2f}%)")
print(f"   TPOT R²:     {r2_tpot:.4f} ({r2_tpot*100:.2f}%)")
print(f"   Improvement: {(r2_tpot - r2_baseline)*100:.2f}%")

print(f"\n🎯 TARGET ACHIEVEMENT:")
if r2_tpot >= 0.80:
    print(f"   ✅ SUCCESS! Exceeded 80% accuracy target")
    print(f"   Achieved: {r2_tpot*100:.2f}%")
else:
    print(f"   ⚠️  Close to target: {r2_tpot*100:.2f}%")
    print(f"   Need: {(0.80-r2_tpot)*100:.2f}% more")

print(f"\n🔬 DATA QUALITY:")
print(f"   Total samples: {len(df_clean)}")
print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Features engineered: {len(feature_columns)}")

print(f"\n📈 RECOMMENDATIONS FOR FURTHER IMPROVEMENT:")
print("""
   1. ✅ MAXIMUM TPOT (150 gen, 100 pop, 480 min) - Already configured!
   2. ✅ This is the most thorough configuration possible!
   3. If results still need improvement, consider:
      • Ensemble methods (stacking multiple TPOT runs with different seeds)
      • Add more chemical descriptors (molecular weight, rotatable bonds,
        TPSA, aromatic rings, H-bond donors/acceptors, LogS, etc.)
      • Use RDKit to calculate 200+ molecular descriptors from SMILES
      • Collect more training data (more samples = better generalization)
   4. Try different search spaces after this run:
      • 'gradient-boosting' for tree-based models
      • 'neural-network' for deep learning approaches
   5. Investigate systematic errors in predictions
   6. Consider domain-specific corrections based on chemical classes
""")

print("="*80)
print("✅ Analysis complete! All results and models saved.")
print("="*80)

print("\n📁 OUTPUT FILES CREATED:")
print("   • logP_prediction_results.png - Visualization plots")
print("   • feature_importance.png - Feature importance chart (if available)")
print("   • tpot_logP_pipeline.py - Exportable Python pipeline")
print("   • tpot_logP_model.pkl - Trained model (pickle)")
print("   • logP_predictions_detailed.csv - All predictions with errors")

print("\n" + "="*80)
print("Script execution completed successfully!")
print("="*80)
