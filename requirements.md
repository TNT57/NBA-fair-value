# requirements.md — Rubric Decoded

> This file translates the marking criteria into exactly what you need to do.
> Read this before writing a single line of code or a single slide.

---

## Marking Breakdown

| Criterion | Weight | What it really means |
|---|---|---|
| Timing: 9–11 minutes | Hard penalty | Practice with a timer. Non-negotiable. |
| Completeness, clarity, accuracy | 50% | Every required section present, correct, logical |
| Slide & presentation quality | 30% | Clean slides, no typos, conversational pace, dot points not paragraphs |
| Attendance both weeks | 10% | Show up to both presentation sessions |
| Deep understanding + insight | 10% | Go beyond plug-and-play. Discuss WHY. |

---

## The 10 Required Sections — and What to Say in Each

### 1. Title Slide
- Your name + student ID
- Assignment title
- Date

### 2. Declaration
**Exact language to use:**
> "This is my own original work. I used [Claude / ChatGPT / GitHub Copilot] to assist with code generation and analysis. I understand all methods, processes, and results presented, and can explain them independently."

If you didn't use GenAI: "This is entirely my own original work."

### 3. Introduction to the Problem
**What the marker wants:** A real problem, clearly explained to a non-technical client AND technically accurate.

**What to cover:**
- NBA salary cap is a hard financial constraint (~$136M in 2023–24 season)
- Teams that overpay one player have less budget for others
- Current valuation is subjective — based on reputation, highlights, agent negotiation
- Problem: there is no systematic, data-driven way to flag salary anomalies
- Stakes: a bad max contract (4–5 years, $40–50M/year) can set a franchise back half a decade

**What NOT to do:** Don't just say "we want to predict salary." That's regression, not your project.

### 4. Map to Data Mining Problem Type
**Your answer:** This is an **outlier detection** problem.

**Why — and you must say all of this:**
- There are NO class labels (we don't have a pre-labelled set of "overpaid" and "underpaid" players)
- We cannot use supervised classification — the truth is unknown
- We cannot use regression — we are not predicting salary, we are detecting anomalies IN the salary-performance relationship
- Outlier detection is appropriate because we want to find players who deviate significantly from the statistical norm
- Isolation Forest is chosen because: (a) it is designed for high-dimensional tabular data, (b) it is computationally efficient, (c) it does not assume a normal distribution, (d) it outperforms distance-based methods (like LOF) on large datasets

### 5. Data Description
Cover:
- Source: Kaggle — "NBA Player Salaries (2022–23 Season)" by jamiewelsh2
- Number of rows (players) and columns (features)
- Key features: Salary, PTS (points), AST (assists), TRB (rebounds), WS (win shares), PER (player efficiency rating), MP (minutes played)
- Data summary: min, max, mean salary; distribution shape (right-skewed)
- Preprocessing steps you did: dropped low-minute players (<500 min), log-transformed salary, standardised features

### 6. Method Introduction
**Isolation Forest — explain it so your peers understand:**

Core idea: Normal points need many random cuts to isolate. Anomalies are isolated quickly.

Step by step:
1. Randomly select a feature
2. Randomly select a split value between min and max of that feature
3. Recursively partition the data
4. Count how many splits it took to isolate each point
5. Average this across many trees — short path = anomaly, long path = normal

Key parameters to explain:
- `n_estimators`: number of trees (use 200)
- `contamination`: expected proportion of outliers (use 0.1 = 10%)
- `max_samples`: samples per tree (use 'auto')

Why NOT other methods:
- K-Means: banned + requires knowing number of clusters
- LOF: distance-based, struggles with high-dimensional data and varying densities
- One-Class SVM: sensitive to hyperparameters, slow on large data

### 7. Procedure
Step by step — what you actually did:
1. Downloaded and merged salary + stats dataset
2. Filtered players with <500 minutes (too small a sample)
3. Engineered features: salary-per-win-share, points-per-dollar, PER/salary ratio
4. Log-transformed salary (right-skewed distribution)
5. Standardised all features with StandardScaler
6. Trained Isolation Forest (contamination=0.1, n_estimators=200)
7. Compared with LOF (n_neighbors=20)
8. Visualised results: scatter plot of Win Shares vs Salary, coloured by anomaly score

### 8. Results and Discussion
Must include:
- List of top 5 flagged overpaid players (high salary, low win shares)
- List of top 5 flagged underpaid players (low salary, high win shares)
- Scatter plot with outliers highlighted
- Comparison of Isolation Forest vs LOF — which found more meaningful outliers and why
- Discussion: Do the flagged players make intuitive sense? Why might a player be overpaid? (injury history, decline, positional scarcity, star power)
- Limitations: dataset is one season only; salary reflects future potential not just past performance; does not account for leadership or marketability

### 9. Conclusion
- Isolation Forest successfully identified statistically anomalous salary cases
- The model provides a starting point for data-driven roster analysis
- Future work: multi-season data, include injury history, add positional context

### 10. References
- Dataset citation (Kaggle link + author)
- Liu, F.T., Ting, K.M., Zhou, Z.H. (2008). Isolation Forest. ICDM.
- Any other papers/sources you referenced

---

## The 10% Insight Mark — How to Get It

Your examiner will award these marks if you show you went BEYOND plug-and-play. Demonstrate:

1. **Deliberate feature engineering** — you didn't just feed raw stats. You created salary-per-win-share ratios that make the outlier detection more meaningful.
2. **Parameter justification** — explain WHY contamination=0.1 (roughly 10% of NBA players are genuinely misvalued at any time — cite the sports analytics literature).
3. **Method comparison** — you ran LOF too, and you can explain why Isolation Forest gave more actionable results.
4. **Real-world interpretation** — don't just say "player X is an outlier." Say "player X is flagged as overpaid, which aligns with their declining PER over 3 seasons and recent injury history."
5. **Honest limitations** — showing you understand what the model CAN'T do is a sign of sophistication, not weakness.
