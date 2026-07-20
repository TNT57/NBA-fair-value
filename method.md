# method.md — Understanding Isolation Forest

> Read this BEFORE you write any code. You need to understand what the algorithm
> is doing, not just run it. Your professor may ask you to explain it.

---

## The Core Intuition (explain this in your presentation)

Imagine you are in a forest trying to find the one person who doesn't belong.

Normal people blend into the crowd — you need many questions to identify them.
Anomalies stand out — you isolate them in very few questions.

Isolation Forest exploits this idea mathematically.

---

## How It Actually Works (step by step)

### Step 1: Build many random trees
- The algorithm builds `n_estimators` isolation trees (default: 100, you'll use 200)
- Each tree is built on a random sample of your data

### Step 2: For each tree, randomly partition the data
- Pick a feature at random (e.g. Salary)
- Pick a random split value between the min and max of that feature
- Divide data into left (below split) and right (above split)
- Repeat recursively until each point is isolated (alone in its partition)

### Step 3: Measure the path length
- Count how many splits it took to isolate each data point
- **Short path = anomaly** (isolated quickly = stands out)
- **Long path = normal** (took many splits to separate from the crowd)

### Step 4: Average across all trees
- Average the path length for each point across all trees
- Convert to an anomaly score (lower score = more anomalous)
- Points below a threshold (controlled by `contamination`) are labelled as outliers

---

## Key Parameters — Know These

| Parameter | What it does | What you'll use | Why |
|---|---|---|---|
| `n_estimators` | Number of trees | 200 | More trees = more stable results |
| `contamination` | Expected % of outliers | 0.10 | ~10% of NBA players are likely misvalued |
| `max_samples` | Data points per tree | 'auto' (256) | Default works well |
| `random_state` | Reproducibility | 42 | So results are the same each run |

---

## What the Output Means

```python
labels = model.predict(X)
#  1 = normal player (salary matches performance)
# -1 = outlier (salary is anomalous vs performance)

scores = model.decision_function(X)
# More negative = more anomalous
# Around 0 = borderline
# More positive = clearly normal
```

---

## Why Isolation Forest for THIS Problem

| Requirement | Does IF meet it? |
|---|---|
| No class labels needed | ✅ Unsupervised — perfect since we don't know "ground truth" overpaid |
| Works on tabular data | ✅ Designed for it |
| Handles right-skewed salary distribution | ✅ Doesn't assume normality (unlike z-score methods) |
| Scales well to 400+ players | ✅ Linear time complexity |
| Interpretable output | ✅ Anomaly scores are rankable |

---

## Why NOT Other Methods

### Why not LOF (Local Outlier Factor)?
- LOF is distance-based — it compares each point to its k nearest neighbours
- Struggles when data has clusters of varying density (NBA data does — stars cluster differently from role players)
- Slower on larger datasets
- BUT: you will run it as a comparison and discuss why IF is better
- LOF is useful for finding LOCAL outliers; IF is better for GLOBAL anomalies

### Why not One-Class SVM?
- Requires careful kernel and hyperparameter tuning
- Sensitive to outliers in training data (circular problem)
- Computationally expensive

### Why not Z-score / IQR method?
- Only works on one variable at a time (univariate)
- Assumes normal distribution (salary is log-normal, not normal)
- Doesn't capture multivariate relationships (e.g. high salary + low WS + low PER together)
- Isolation Forest captures ALL features simultaneously — that's the advantage

---

## The Feature Engineering Rationale

Raw features alone are not enough. A player earning $30M and scoring 25 PPG looks fine.
But if their Win Shares is 2.0 (terrible efficiency), that context matters.

Your engineered features:

### salary_per_WS = Salary / (Win Shares + 0.1)
- Win Shares measures how many team wins a player contributed to
- High salary_per_WS = paying a lot for few wins = potential overpay
- +0.1 avoids division by zero for players with 0 or negative WS

### pts_per_dollar = PTS / (Salary in millions)
- Raw scoring efficiency relative to cost
- A player scoring 20 PPG on $3M is extremely good value
- A player scoring 12 PPG on $35M is a concern

### PER_salary_ratio = PER / (Salary in millions)
- PER (Player Efficiency Rating) is a composite all-around performance metric
- Average NBA PER = 15.0
- High ratio = efficient player being paid relatively little
- Low ratio = inefficient player being paid a lot

---

## How to Explain This in 2 Minutes During Your Presentation

> "Isolation Forest works by building hundreds of random decision trees. Each tree tries
> to isolate individual data points through random splits. The key insight is that
> anomalies — like a player being paid $40 million while contributing almost nothing
> statistically — are isolated in very few splits. Normal players blend in and take
> many splits to separate. The algorithm assigns each player an anomaly score based
> on the average number of splits needed across all trees. The players with the
> shortest average path — the ones isolated quickest — are flagged as outliers.
> In our case, those are the players whose salary is most out of line with
> their actual on-court contribution."
