# code_guide.md — Full Python Code Walkthrough

> Copy this into your IDE (VSCode / Jupyter). Each section is labelled.
> Claude Code in VSCode can help you debug and extend any section.

---

## Section 0: Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# Reproducibility
np.random.seed(42)

# Plot style
sns.set_theme(style="whitegrid", palette="muted")
```

---

## Section 1: Load and Inspect Data

```python
# Load the Kaggle dataset
df = pd.read_csv('data/nba_salaries_2022_23.csv')

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nSalary statistics:")
print(df['Salary'].describe())
```

---

## Section 2: Data Cleaning

```python
# Check column names — adjust these to match your actual CSV
# Common column names in this dataset:
# 'Player', 'Salary', 'PTS', 'AST', 'TRB', 'WS', 'PER', 'MP', 'G', 'Pos'

# Drop duplicates (some players appear on multiple teams)
# Keep the row where Team == 'TOT' (total stats for traded players)
if 'Tm' in df.columns:
    df = df.sort_values('Tm').drop_duplicates(subset='Player', keep='last')

# Filter: minimum 500 minutes played
# Rationale: players with very few minutes have unreliable per-game stats
df = df[df['MP'] >= 500].copy()

print(f"Players after filtering: {len(df)}")

# Drop rows with missing salary or key stats
key_cols = ['Salary', 'PTS', 'AST', 'TRB', 'WS', 'PER']
df = df.dropna(subset=key_cols)
print(f"Players after dropping NaN: {len(df)}")

# Reset index
df = df.reset_index(drop=True)
```

---

## Section 3: Exploratory Data Analysis

```python
# --- Plot 1: Salary Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df['Salary'] / 1e6, bins=30, color='steelblue', edgecolor='white')
axes[0].set_title('Salary Distribution (Raw)', fontsize=13)
axes[0].set_xlabel('Salary ($M)')
axes[0].set_ylabel('Count')

axes[1].hist(np.log(df['Salary']), bins=30, color='coral', edgecolor='white')
axes[1].set_title('Salary Distribution (Log-transformed)', fontsize=13)
axes[1].set_xlabel('Log(Salary)')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('outputs/salary_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Plot 2: Salary vs Win Shares ---
plt.figure(figsize=(8, 5))
plt.scatter(df['WS'], df['Salary'] / 1e6, alpha=0.5, color='steelblue', s=40)
plt.xlabel('Win Shares', fontsize=12)
plt.ylabel('Salary ($M)', fontsize=12)
plt.title('NBA 2022-23: Win Shares vs Salary', fontsize=13)
plt.savefig('outputs/ws_vs_salary_raw.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Correlation matrix ---
corr_cols = ['Salary', 'PTS', 'AST', 'TRB', 'WS', 'PER', 'MP']
corr = df[corr_cols].corr()
plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('outputs/correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Section 4: Feature Engineering

```python
# Engineered features that make the outlier detection more meaningful
# These capture the RELATIONSHIP between salary and performance

salary_m = df['Salary'] / 1_000_000  # salary in millions for readability

df['salary_per_WS'] = df['Salary'] / (df['WS'] + 0.1)   # cost per win contributed
df['pts_per_dollar'] = df['PTS'] / (salary_m + 0.01)      # scoring efficiency per $M
df['PER_salary_ratio'] = df['PER'] / (salary_m + 0.01)    # overall efficiency per $M
df['log_salary'] = np.log(df['Salary'])                    # normalise salary distribution

print("Engineered features added:")
print(df[['Player', 'Salary', 'WS', 'PER', 'salary_per_WS', 'pts_per_dollar', 'PER_salary_ratio']].head(10))
```

---

## Section 5: Prepare Feature Matrix

```python
# Select features for the model
# We include both raw performance stats AND engineered ratio features
feature_cols = [
    'log_salary',        # normalised salary
    'PTS',               # points per game
    'AST',               # assists per game
    'TRB',               # rebounds per game
    'WS',                # win shares (season total)
    'PER',               # player efficiency rating
    'salary_per_WS',     # cost per win — key ratio
    'pts_per_dollar',    # scoring per dollar
    'PER_salary_ratio',  # overall efficiency per dollar
]

X = df[feature_cols].copy()

# Standardise (zero mean, unit variance)
# Required because features have very different scales
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

print("Feature matrix shape:", X_scaled.shape)
print("Scaled feature stats:")
print(X_scaled_df.describe().round(2))
```

---

## Section 6: Isolation Forest

```python
# --- Train Isolation Forest ---
# contamination=0.10 means we expect ~10% of players to be salary anomalies
# This is a reasonable assumption for professional sports

IF_model = IsolationForest(
    n_estimators=200,
    contamination=0.10,
    max_samples='auto',
    random_state=42
)

IF_model.fit(X_scaled)

# Get predictions and anomaly scores
df['IF_label'] = IF_model.predict(X_scaled)        # 1=normal, -1=outlier
df['IF_score'] = IF_model.decision_function(X_scaled)  # lower = more anomalous

# How many outliers flagged?
n_outliers = (df['IF_label'] == -1).sum()
print(f"\nIsolation Forest flagged {n_outliers} outliers ({n_outliers/len(df)*100:.1f}%)")

# --- Most anomalous (overpaid candidates) ---
# High salary, low performance → very negative score
overpaid = df[df['IF_label'] == -1].sort_values('IF_score').head(10)
print("\nTop Flagged Players (most anomalous):")
print(overpaid[['Player', 'Salary', 'PTS', 'WS', 'PER', 'IF_score']].to_string())
```

---

## Section 7: Sensitivity Analysis (Contamination Parameter)

```python
# Try different contamination values — show you didn't just use the default
results = {}
for c in [0.05, 0.10, 0.15, 0.20]:
    m = IsolationForest(n_estimators=200, contamination=c, random_state=42)
    m.fit(X_scaled)
    labels = m.predict(X_scaled)
    results[c] = {
        'n_outliers': (labels == -1).sum(),
        'outlier_pct': (labels == -1).mean() * 100
    }

print("\nContamination Sensitivity Analysis:")
for c, r in results.items():
    print(f"  contamination={c:.2f} → {r['n_outliers']} outliers ({r['outlier_pct']:.1f}%)")
```

---

## Section 8: LOF Comparison

```python
# Local Outlier Factor for comparison
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.10
)

df['LOF_label'] = lof.fit_predict(X_scaled)  # 1=normal, -1=outlier

# Agreement between methods
df['both_flag'] = (df['IF_label'] == -1) & (df['LOF_label'] == -1)
df['IF_only'] = (df['IF_label'] == -1) & (df['LOF_label'] == 1)
df['LOF_only'] = (df['IF_label'] == 1) & (df['LOF_label'] == -1)

print("\nMethod Agreement:")
print(f"  Both methods flag as outlier: {df['both_flag'].sum()}")
print(f"  Only Isolation Forest flags:  {df['IF_only'].sum()}")
print(f"  Only LOF flags:               {df['LOF_only'].sum()}")

# Players both methods agree on
consensus_outliers = df[df['both_flag']].sort_values('IF_score')
print("\nConsensus Outliers (flagged by both):")
print(consensus_outliers[['Player', 'Salary', 'PTS', 'WS', 'PER']].to_string())
```

---

## Section 9: Final Visualisation

```python
# --- Main Result Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

colors = df['IF_label'].map({1: '#aec6cf', -1: '#e05c5c'})  # blue=normal, red=outlier

scatter = ax.scatter(
    df['WS'],
    df['Salary'] / 1e6,
    c=colors,
    s=60,
    alpha=0.7,
    edgecolors='white',
    linewidths=0.5
)

# Label the top 8 outliers by name
top_outliers = df[df['IF_label'] == -1].sort_values('IF_score').head(8)
for _, row in top_outliers.iterrows():
    ax.annotate(
        row['Player'].split()[-1],  # last name only to avoid clutter
        (row['WS'], row['Salary'] / 1e6),
        textcoords='offset points',
        xytext=(6, 4),
        fontsize=7.5,
        color='#c0392b'
    )

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#aec6cf', label='Normal (salary matches performance)'),
    Patch(facecolor='#e05c5c', label='Outlier (statistically anomalous salary)')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

ax.set_xlabel('Win Shares (2022–23)', fontsize=12)
ax.set_ylabel('Annual Salary ($M)', fontsize=12)
ax.set_title('NBA 2022–23: Salary Anomaly Detection via Isolation Forest', fontsize=13)

plt.tight_layout()
plt.savefig('outputs/final_result_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to outputs/final_result_plot.png")
```

---

## Section 10: Export Results Table

```python
# Clean summary table for slides
summary = df[['Player', 'Salary', 'PTS', 'AST', 'TRB', 'WS', 'PER', 'IF_label', 'IF_score']].copy()
summary['Salary_M'] = (summary['Salary'] / 1e6).round(1)
summary['Verdict'] = summary['IF_label'].map({1: 'Normal', -1: 'OUTLIER'})
summary = summary.sort_values('IF_score')

# Top overpaid
print("\n=== TOP FLAGGED (MOST ANOMALOUS) ===")
print(summary[summary['Verdict'] == 'OUTLIER'].head(10)[
    ['Player', 'Salary_M', 'PTS', 'WS', 'PER', 'IF_score']
].to_string())

# Save to CSV
summary.to_csv('outputs/results_table.csv', index=False)
print("\nResults saved to outputs/results_table.csv")
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `KeyError: 'WS'` | Column name is different in your CSV | `print(df.columns)` and adjust |
| `ValueError: Input contains NaN` | Missing values in feature matrix | Add `df.dropna(subset=feature_cols)` before scaling |
| All players flagged as outliers | Contamination too high | Lower contamination to 0.05 |
| No outliers flagged | Contamination too low | Raise contamination to 0.15 |
| Division by zero in feature engineering | Player with 0 WS | The `+0.1` guard handles this — check it's there |
