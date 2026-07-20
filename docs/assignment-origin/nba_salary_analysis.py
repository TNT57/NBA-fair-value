# =============================================================================
# NBA Salary Outlier Detection using Isolation Forest
# Assignment 3 — Data & Web Mining
# =============================================================================
# This script performs outlier detection on NBA player salaries using
# Isolation Forest, with LOF as a comparison method.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 70)
print("STEP 1: Loading Data")
print("=" * 70)

df = pd.read_csv('../data/nba_2022-23_all_stats_with_salary.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows preview:")
print(df[['Player Name', 'Salary', 'PTS', 'AST', 'TRB', 'WS', 'PER', 'MP']].head())

# =============================================================================
# 2. EXPLORATORY DATA ANALISYS (EDA)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: Exploratory Data Analysis")
print("=" * 70)

# Basic statistics
print(f"\nSalary Statistics:")
print(f"  Min:    ${df['Salary'].min():,.0f}")
print(f"  Max:    ${df['Salary'].max():,.0f}")
print(f"  Mean:   ${df['Salary'].mean():,.0f}")
print(f"  Median: ${df['Salary'].median():,.0f}")
print(f"  Std:    ${df['Salary'].std():,.0f}")

# Check missing values in key columns
key_cols = ['Salary', 'PTS', 'AST', 'TRB', 'WS', 'PER', 'MP']
print(f"\nMissing values in key columns:")
for col in key_cols:
    print(f"  {col}: {df[col].isna().sum()}")

# --- Plot 1: Salary Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['Salary'] / 1e6, bins=30, color='steelblue', edgecolor='black', alpha=0.8)
axes[0].set_xlabel('Salary ($ Millions)')
axes[0].set_ylabel('Number of Players')
axes[0].set_title('NBA Salary Distribution (2022-23 Season)')
axes[0].axvline(df['Salary'].mean() / 1e6, color='red', linestyle='--', label=f'Mean: ${df["Salary"].mean()/1e6:.1f}M')
axes[0].axvline(df['Salary'].median() / 1e6, color='orange', linestyle='--', label=f'Median: ${df["Salary"].median()/1e6:.1f}M')
axes[0].legend()

# Log-transformed salary
log_salary = np.log(df['Salary'].dropna())
axes[1].hist(log_salary, bins=30, color='darkgreen', edgecolor='black', alpha=0.8)
axes[1].set_xlabel('Log(Salary)')
axes[1].set_ylabel('Number of Players')
axes[1].set_title('Log-Transformed Salary Distribution')
axes[1].axvline(log_salary.mean(), color='red', linestyle='--', label=f'Mean: {log_salary.mean():.2f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('../outputs/01_salary_distribution.png', bbox_inches='tight')
plt.close()
print("\n✓ Saved: outputs/01_salary_distribution.png")

# --- Plot 2: Key Stats Distributions ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].hist(df['PTS'].dropna(), bins=25, color='coral', edgecolor='black', alpha=0.8)
axes[0].set_xlabel('Points Per Game')
axes[0].set_ylabel('Count')
axes[0].set_title('Points Distribution')

axes[1].hist(df['WS'].dropna(), bins=25, color='mediumpurple', edgecolor='black', alpha=0.8)
axes[1].set_xlabel('Win Shares')
axes[1].set_ylabel('Count')
axes[1].set_title('Win Shares Distribution')

axes[2].hist(df['PER'].dropna(), bins=25, color='goldenrod', edgecolor='black', alpha=0.8)
axes[2].set_xlabel('PER')
axes[2].set_ylabel('Count')
axes[2].set_title('PER Distribution')

plt.tight_layout()
plt.savefig('../outputs/02_stats_distributions.png', bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/02_stats_distributions.png")

# --- Plot 3: Salary vs Win Shares scatter ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['WS'], df['Salary'] / 1e6, alpha=0.5, color='steelblue', edgecolors='grey', s=40)
ax.set_xlabel('Win Shares')
ax.set_ylabel('Salary ($ Millions)')
ax.set_title('Salary vs Win Shares (All Players)')
plt.tight_layout()
plt.savefig('../outputs/03_salary_vs_ws_raw.png', bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/03_salary_vs_ws_raw.png")

# --- Plot 4: Correlation heatmap ---
corr_cols = ['Salary', 'PTS', 'AST', 'TRB', 'WS', 'PER', 'MP', 'Age', 'USG%', 'BPM']
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax, 
            square=True, linewidths=0.5)
ax.set_title('Correlation Matrix: Salary vs Performance Metrics')
plt.tight_layout()
plt.savefig('../outputs/04_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/04_correlation_heatmap.png")

# =============================================================================
# 3. PREPROCESSING
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: Preprocessing")
print("=" * 70)

# Filter players with < 500 total minutes
print(f"\nPlayers before filtering: {len(df)}")
print(f"  MP (minutes per game) range: {df['MP'].min():.1f} — {df['MP'].max():.1f}")
print(f"  Total Minutes range: {df['Total Minutes'].min():.0f} — {df['Total Minutes'].max():.0f}")
df_clean = df[df['Total Minutes'] >= 500].copy()
print(f"Players after filtering (Total Minutes >= 500): {len(df_clean)}")

# Drop rows with missing key values
key_features = ['Salary', 'PTS', 'AST', 'TRB', 'WS', 'PER', 'MP', 'USG%', 'BPM', 'VORP']
df_clean = df_clean.dropna(subset=key_features)
print(f"Players after dropping NaN in key features: {len(df_clean)}")

# Reset index
df_clean = df_clean.reset_index(drop=True)

# --- Feature Engineering ---
print("\nFeature Engineering...")

# Log-transform salary (salary is right-skewed)
df_clean['log_salary'] = np.log(df_clean['Salary'])

# Salary per Win Share (use abs to avoid negative denominator issues)
df_clean['salary_per_WS'] = df_clean['Salary'] / (df_clean['WS'].abs() + 1.0)

# Points per million dollars
df_clean['pts_per_dollar'] = df_clean['PTS'] / (df_clean['Salary'] / 1_000_000)

# PER to salary ratio
df_clean['PER_salary_ratio'] = df_clean['PER'] / (df_clean['Salary'] / 1_000_000)

# Replace any remaining infinities with large finite values
df_clean['salary_per_WS'] = df_clean['salary_per_WS'].replace([np.inf, -np.inf], df_clean['salary_per_WS'].replace([np.inf, -np.inf], np.nan).max())
df_clean['pts_per_dollar'] = df_clean['pts_per_dollar'].replace([np.inf, -np.inf], 0)
df_clean['PER_salary_ratio'] = df_clean['PER_salary_ratio'].replace([np.inf, -np.inf], 0)

print(f"  salary_per_WS:  min={df_clean['salary_per_WS'].min():,.0f}, max={df_clean['salary_per_WS'].max():,.0f}")
print(f"  pts_per_dollar: min={df_clean['pts_per_dollar'].min():.4f}, max={df_clean['pts_per_dollar'].max():.4f}")
print(f"  PER_salary_ratio: min={df_clean['PER_salary_ratio'].min():.6f}, max={df_clean['PER_salary_ratio'].max():.6f}")

# --- Feature Matrix ---
# Use log_salary instead of raw salary, plus performance metrics and engineered features
feature_cols = ['log_salary', 'PTS', 'AST', 'TRB', 'WS', 'PER', 'USG%', 'BPM', 'VORP',
                'salary_per_WS', 'pts_per_dollar', 'PER_salary_ratio']

X = df_clean[feature_cols].copy()
print(f"\nFeature matrix shape: {X.shape}")
print(f"Features used: {feature_cols}")

# Standardise all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
print("✓ Features standardised with StandardScaler")

# =============================================================================
# 4. ISOLATION FOREST MODEL
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: Isolation Forest Modelling")
print("=" * 70)

# Train Isolation Forest with contamination=0.1
print("\nTraining Isolation Forest (n_estimators=200, contamination=0.1)...")
iso_forest = IsolationForest(n_estimators=200, contamination=0.1, max_samples='auto', random_state=42)
iso_forest.fit(X_scaled)

# Get predictions and scores
df_clean['IF_label'] = iso_forest.predict(X_scaled)      # 1=normal, -1=outlier
df_clean['IF_score'] = iso_forest.decision_function(X_scaled)  # lower = more anomalous

n_outliers = (df_clean['IF_label'] == -1).sum()
n_normal = (df_clean['IF_label'] == 1).sum()
print(f"  Normal players: {n_normal}")
print(f"  Outlier players: {n_outliers}")

# --- Try contamination=0.05 and 0.15 for comparison ---
print("\nComparing contamination values...")
for c in [0.05, 0.15]:
    temp_model = IsolationForest(n_estimators=200, contamination=c, random_state=42)
    temp_labels = temp_model.fit_predict(X_scaled)
    n_out = (temp_labels == -1).sum()
    print(f"  contamination={c}: {n_out} outliers flagged")

# =============================================================================
# 5. LOF COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: LOF Comparison")
print("=" * 70)

print("Training Local Outlier Factor (n_neighbors=20, contamination=0.1)...")
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
lof_labels = lof.fit_predict(X_scaled)
lof_scores = lof.negative_outlier_factor_

df_clean['LOF_label'] = lof_labels
df_clean['LOF_score'] = lof_scores  # more negative = more anomalous

n_lof_outliers = (lof_labels == -1).sum()
print(f"  LOF outliers flagged: {n_lof_outliers}")

# Compare agreement between IF and LOF
both_outlier = ((df_clean['IF_label'] == -1) & (df_clean['LOF_label'] == -1)).sum()
if_only = ((df_clean['IF_label'] == -1) & (df_clean['LOF_label'] == 1)).sum()
lof_only = ((df_clean['IF_label'] == 1) & (df_clean['LOF_label'] == -1)).sum()
both_normal = ((df_clean['IF_label'] == 1) & (df_clean['LOF_label'] == 1)).sum()

print(f"\nAgreement between IF and LOF:")
print(f"  Both flag as outlier:   {both_outlier}")
print(f"  Only IF flags outlier:  {if_only}")
print(f"  Only LOF flags outlier: {lof_only}")
print(f"  Both flag as normal:    {both_normal}")

# =============================================================================
# 6. IDENTIFY TOP OVERPAID AND UNDERPAID
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: Identifying Overpaid and Underpaid Players")
print("=" * 70)

# Outliers flagged by Isolation Forest
outliers = df_clean[df_clean['IF_label'] == -1].copy()
print(f"\nTotal IF outliers: {len(outliers)}")

# --- Intuitive classification using salary_per_WS ---
# salary_per_WS = how much teams pay per unit of win production
#   HIGH salary_per_WS → paying a lot for little production → OVERPAID
#   LOW  salary_per_WS → paying little for great production → UNDERPAID
# This is more intuitive than splitting by median salary because
# Isolation Forest flags ALL types of anomalies (extreme performers too),
# not just "overpaid" vs "underpaid".

# Overpaid: sort outliers by salary_per_WS descending (worst value first)
overpaid = outliers.sort_values('salary_per_WS', ascending=False)
print("\n🏆 TOP 5 FLAGGED OVERPAID PLAYERS (Highest Salary per Win Share):")
print("-" * 70)
for i, (_, row) in enumerate(overpaid.head(5).iterrows()):
    print(f"  {i+1}. {row['Player Name']:<25s} | Salary: ${row['Salary']/1e6:>6.1f}M | "
          f"WS: {row['WS']:>5.1f} | PER: {row['PER']:>5.1f} | PTS: {row['PTS']:>5.1f} | "
          f"$M/WS: {row['salary_per_WS']/1e6:>6.1f}")

# Underpaid: sort outliers by salary_per_WS ascending (best value first)
underpaid = outliers.sort_values('salary_per_WS', ascending=True)
print("\n💎 TOP 5 FLAGGED UNDERPAID PLAYERS (Lowest Salary per Win Share):")
print("-" * 70)
for i, (_, row) in enumerate(underpaid.head(5).iterrows()):
    print(f"  {i+1}. {row['Player Name']:<25s} | Salary: ${row['Salary']/1e6:>6.1f}M | "
          f"WS: {row['WS']:>5.1f} | PER: {row['PER']:>5.1f} | PTS: {row['PTS']:>5.1f} | "
          f"$M/WS: {row['salary_per_WS']/1e6:>6.1f}")

# --- LOF Top outliers for comparison ---
lof_outliers = df_clean[df_clean['LOF_label'] == -1].sort_values('LOF_score')
print("\n📋 TOP 5 LOF OUTLIERS:")
print("-" * 70)
for i, (_, row) in enumerate(lof_outliers.head(5).iterrows()):
    print(f"  {i+1}. {row['Player Name']:<25s} | Salary: ${row['Salary']/1e6:>6.1f}M | "
          f"WS: {row['WS']:>5.1f} | PER: {row['PER']:>5.1f} | "
          f"LOF Score: {row['LOF_score']:>6.3f}")

# =============================================================================
# 7. VISUALISATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: Generating Visualisations")
print("=" * 70)

# --- Plot 5: WS vs Salary colored by IF anomaly label ---
fig, ax = plt.subplots(figsize=(12, 7))
normal = df_clean[df_clean['IF_label'] == 1]
anomaly = df_clean[df_clean['IF_label'] == -1]

ax.scatter(normal['WS'], normal['Salary'] / 1e6, c='lightgrey', alpha=0.6, s=40, label='Normal', edgecolors='grey')
ax.scatter(anomaly['WS'], anomaly['Salary'] / 1e6, c='red', alpha=0.8, s=60, label='Outlier (IF)', edgecolors='darkred')

# Label top 5 most anomalous
top5_anom = anomaly.sort_values('IF_score').head(5)
for _, row in top5_anom.iterrows():
    ax.annotate(row['Player Name'], (row['WS'], row['Salary'] / 1e6),
                fontsize=8, fontweight='bold', 
                xytext=(5, 5), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=0.8))

ax.set_xlabel('Win Shares', fontsize=13)
ax.set_ylabel('Salary ($ Millions)', fontsize=13)
ax.set_title('Isolation Forest: Win Shares vs Salary (Red = Outlier)', fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('../outputs/05_IF_ws_vs_salary.png', bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/05_IF_ws_vs_salary.png")

# --- Plot 6: PER vs Salary colored by IF label ---
fig, ax = plt.subplots(figsize=(12, 7))
ax.scatter(normal['PER'], normal['Salary'] / 1e6, c='lightgrey', alpha=0.6, s=40, label='Normal', edgecolors='grey')
ax.scatter(anomaly['PER'], anomaly['Salary'] / 1e6, c='red', alpha=0.8, s=60, label='Outlier (IF)', edgecolors='darkred')

for _, row in top5_anom.iterrows():
    ax.annotate(row['Player Name'], (row['PER'], row['Salary'] / 1e6),
                fontsize=8, fontweight='bold',
                xytext=(5, 5), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=0.8))

ax.set_xlabel('PER (Player Efficiency Rating)', fontsize=13)
ax.set_ylabel('Salary ($ Millions)', fontsize=13)
ax.set_title('Isolation Forest: PER vs Salary (Red = Outlier)', fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('../outputs/06_IF_per_vs_salary.png', bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/06_IF_per_vs_salary.png")

# --- Plot 7: IF vs LOF comparison ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# IF plot
normal_if = df_clean[df_clean['IF_label'] == 1]
outlier_if = df_clean[df_clean['IF_label'] == -1]
axes[0].scatter(normal_if['WS'], normal_if['Salary'] / 1e6, c='lightgrey', alpha=0.5, s=30)
axes[0].scatter(outlier_if['WS'], outlier_if['Salary'] / 1e6, c='red', alpha=0.7, s=50, edgecolors='darkred')
axes[0].set_xlabel('Win Shares')
axes[0].set_ylabel('Salary ($ Millions)')
axes[0].set_title(f'Isolation Forest ({len(outlier_if)} outliers)')

# LOF plot
normal_lof = df_clean[df_clean['LOF_label'] == 1]
outlier_lof = df_clean[df_clean['LOF_label'] == -1]
axes[1].scatter(normal_lof['WS'], normal_lof['Salary'] / 1e6, c='lightgrey', alpha=0.5, s=30)
axes[1].scatter(outlier_lof['WS'], outlier_lof['Salary'] / 1e6, c='blue', alpha=0.7, s=50, edgecolors='darkblue')
axes[1].set_xlabel('Win Shares')
axes[1].set_ylabel('Salary ($ Millions)')
axes[1].set_title(f'LOF ({len(outlier_lof)} outliers)')

fig.suptitle('Isolation Forest vs LOF: Outlier Detection Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/07_IF_vs_LOF_comparison.png', bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/07_IF_vs_LOF_comparison.png")

# --- Plot 8: Anomaly Score Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df_clean['IF_score'], bins=30, color='steelblue', edgecolor='black', alpha=0.8)
axes[0].axvline(0, color='red', linestyle='--', label='Decision boundary')
axes[0].set_xlabel('IF Anomaly Score')
axes[0].set_ylabel('Count')
axes[0].set_title('Isolation Forest Score Distribution')
axes[0].legend()

axes[1].hist(df_clean['LOF_score'], bins=30, color='darkorange', edgecolor='black', alpha=0.8)
axes[1].set_xlabel('LOF Score (negative = more anomalous)')
axes[1].set_ylabel('Count')
axes[1].set_title('LOF Score Distribution')

plt.tight_layout()
plt.savefig('../outputs/08_anomaly_score_distribution.png', bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/08_anomaly_score_distribution.png")

# --- Plot 9: Salary per Win Share (engineered feature) ---
fig, ax = plt.subplots(figsize=(12, 7))
scatter = ax.scatter(df_clean['WS'], df_clean['salary_per_WS'] / 1e6, 
                     c=df_clean['IF_label'].map({1: 'grey', -1: 'red'}),
                     alpha=0.6, s=40, edgecolors='black', linewidth=0.3)
ax.set_xlabel('Win Shares')
ax.set_ylabel('Salary per Win Share ($ Millions)')
ax.set_title('Salary per Win Share — Highlighting Inefficiencies')
ax.axhline(df_clean['salary_per_WS'].median() / 1e6, color='blue', linestyle='--', 
           alpha=0.5, label=f'Median: ${df_clean["salary_per_WS"].median()/1e6:.1f}M/WS')
ax.legend()
plt.tight_layout()
plt.savefig('../outputs/09_salary_per_ws.png', bbox_inches='tight')
plt.close()
print("✓ Saved: outputs/09_salary_per_ws.png")

# =============================================================================
# 8. SAVE RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: Saving Results")
print("=" * 70)

# Save full results to CSV
output_cols = ['Player Name', 'Salary', 'Position', 'Age', 'Team', 'GP', 'MP',
               'PTS', 'AST', 'TRB', 'WS', 'PER', 'USG%', 'BPM', 'VORP',
               'log_salary', 'salary_per_WS', 'pts_per_dollar', 'PER_salary_ratio',
               'IF_label', 'IF_score', 'LOF_label', 'LOF_score']
df_clean[output_cols].to_csv('../outputs/nba_outlier_results.csv', index=False)
print("✓ Saved: outputs/nba_outlier_results.csv")

# Save summary text file
with open('../outputs/analysis_summary.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("NBA SALARY OUTLIER DETECTION — ANALYSIS SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("DATASET OVERVIEW\n")
    f.write("-" * 40 + "\n")
    f.write(f"Original dataset: {df.shape[0]} players, {df.shape[1]} features\n")
    f.write(f"After filtering (MP >= 500): {len(df_clean)} players\n")
    f.write(f"Features used: {len(feature_cols)}\n\n")
    
    f.write("SALARY STATISTICS\n")
    f.write("-" * 40 + "\n")
    f.write(f"Min salary:    ${df_clean['Salary'].min():,.0f}\n")
    f.write(f"Max salary:    ${df_clean['Salary'].max():,.0f}\n")
    f.write(f"Mean salary:   ${df_clean['Salary'].mean():,.0f}\n")
    f.write(f"Median salary: ${df_clean['Salary'].median():,.0f}\n\n")
    
    f.write("ISOLATION FOREST RESULTS\n")
    f.write("-" * 40 + "\n")
    f.write(f"Parameters: n_estimators=200, contamination=0.1, random_state=42\n")
    f.write(f"Normal players: {(df_clean['IF_label']==1).sum()}\n")
    f.write(f"Outlier players: {(df_clean['IF_label']==-1).sum()}\n\n")
    
    f.write("TOP 5 OVERPAID PLAYERS (High Salary, Low Performance)\n")
    f.write("-" * 40 + "\n")
    for i, (_, row) in enumerate(overpaid.head(5).iterrows()):
        f.write(f"  {i+1}. {row['Player Name']:<20s} Salary: ${row['Salary']/1e6:.1f}M  "
                f"WS: {row['WS']:.1f}  PER: {row['PER']:.1f}  PTS: {row['PTS']:.1f}\n")
    f.write("\n")
    
    f.write("TOP 5 UNDERPAID PLAYERS (Low Salary, High Performance)\n")
    f.write("-" * 40 + "\n")
    for i, (_, row) in enumerate(underpaid.head(5).iterrows()):
        f.write(f"  {i+1}. {row['Player Name']:<20s} Salary: ${row['Salary']/1e6:.1f}M  "
                f"WS: {row['WS']:.1f}  PER: {row['PER']:.1f}  PTS: {row['PTS']:.1f}\n")
    f.write("\n")
    
    f.write("LOF COMPARISON\n")
    f.write("-" * 40 + "\n")
    f.write(f"Parameters: n_neighbors=20, contamination=0.1\n")
    f.write(f"LOF outliers flagged: {n_lof_outliers}\n")
    f.write(f"Both methods agree (outlier): {both_outlier}\n")
    f.write(f"Only IF flags outlier: {if_only}\n")
    f.write(f"Only LOF flags outlier: {lof_only}\n\n")
    
    f.write("TOP 5 LOF OUTLIERS\n")
    f.write("-" * 40 + "\n")
    for i, (_, row) in enumerate(lof_outliers.head(5).iterrows()):
        f.write(f"  {i+1}. {row['Player Name']:<20s} Salary: ${row['Salary']/1e6:.1f}M  "
                f"WS: {row['WS']:.1f}  PER: {row['PER']:.1f}\n")

print("✓ Saved: outputs/analysis_summary.txt")

print("\n" + "=" * 70)
print("ALL DONE! Analysis complete.")
print("=" * 70)
print(f"\nGenerated files in outputs/:")
print(f"  01_salary_distribution.png")
print(f"  02_stats_distributions.png")
print(f"  03_salary_vs_ws_raw.png")
print(f"  04_correlation_heatmap.png")
print(f"  05_IF_ws_vs_salary.png")
print(f"  06_IF_per_vs_salary.png")
print(f"  07_IF_vs_LOF_comparison.png")
print(f"  08_anomaly_score_distribution.png")
print(f"  09_salary_per_ws.png")
print(f"  nba_outlier_results.csv")
print(f"  analysis_summary.txt")