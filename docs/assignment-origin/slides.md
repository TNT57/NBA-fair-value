# slides.md — Slide-by-Slide Content Guide

> Use this as your blueprint when building the PowerPoint.
> All content below goes in as TYPED TEXT (dot points), not images.
> Each slide should take roughly 50–70 seconds to present.

---

## Design Rules (read before building)
- Font: Calibri or Arial, 24pt body, 32pt titles
- Colour scheme: Dark navy (#1a1a2e) background + white text, OR clean white + dark text — pick one
- Max 6 dot points per slide
- Each dot point: 1 line, not a sentence fragment, not a paragraph
- Charts go on their own slide (counts as image allocation — use sparingly)
- No animations

---

## Slide 1: Title

**[Title]** Detecting NBA Salary Anomalies Using Isolation Forest

**[Subtitle]** A Data Mining Approach to Identifying Overpaid and Underpaid Players

**[Your name + Student ID]**
**[Course name + Date]**

*Presenter notes: Just introduce yourself and what the talk is about. 10 seconds.*

---

## Slide 2: Declaration

**[Title]** Declaration

**[Body — dot points]**
- This is my own individual work
- GenAI (Claude) was used to assist with code generation and analysis planning
- All methods, processes, and results are understood and can be explained independently
- Dataset sourced from Kaggle — full citation in References slide

*Presenter notes: Read this briefly. Don't linger. Move on.*

---

## Slide 3: The Problem

**[Title]** The Problem — NBA Salary Misallocation

**[Body — dot points]**
- NBA teams operate under a hard salary cap (~$136M in 2022–23 season)
- Every dollar overspent on one player reduces budget for the rest of the roster
- Player salaries are negotiated based on reputation, highlights, and agent leverage — not purely performance data
- A misallocated max contract ($40–50M/year over 4–5 years) can cripple a franchise
- Problem: no systematic, data-driven method exists to flag salary anomalies before signing

**[Optional quote — typed, not image]**
> "We need to find value in places that other teams haven't." — Moneyball (2011)

*Presenter notes: Open with the Ben Simmons example — "$37M, 42 games, $880K per game." Hook the audience before the first slide even ends.*

---

## Slide 4: Data Mining Problem Type

**[Title]** Mapping to a Data Mining Problem

**[Body — dot points]**
- Problem type: **Outlier / Anomaly Detection**
- No class labels exist — we do not have a pre-labelled set of "overpaid" or "underpaid" players
- Supervised classification is not applicable: ground truth is unknown
- Regression is not applicable: we are not predicting salary — we are detecting deviations in the salary-performance relationship
- Outlier detection finds players who are statistically abnormal compared to the rest of the league
- Method chosen: **Isolation Forest** — designed for tabular data, no distribution assumption, scalable

*Presenter notes: Stress the "no labels" point. It's why this MUST be outlier detection, not classification.*

---

## Slide 5: Dataset

**[Title]** Data Description

**[Body — dot points]**
- Source: Kaggle — "NBA Player Salaries (2022–23 Season)" by jamiewelsh2
- 467 player records; filtered to 350 players with ≥ 500 minutes played
- Key features: Salary, Points (PTS), Assists (AST), Rebounds (TRB), Win Shares (WS), Player Efficiency Rating (PER), Minutes Played (MP)
- Salary range: $0.1M (minimum) to $47.6M (maximum); heavily right-skewed → log-transformed
- Engineered features: salary-per-win-share, points-per-dollar, PER-to-salary ratio
- Min-500-minute filter removes players with insufficient sample size to evaluate fairly

**[Chart on this slide or next slide]**
Insert: salary distribution (before/after log transform) — `outputs/salary_distribution.png`

*Presenter notes: Mention why you filtered <500 min players. It shows methodological care.*

---

## Slide 6: Method — Isolation Forest

**[Title]** Method: Isolation Forest

**[Body — dot points]**
- Ensemble anomaly detection algorithm (Liu, Ting & Zhou, 2008)
- Core idea: anomalies are easier to isolate than normal points
- Builds n_estimators random trees; each tree randomly partitions data by splitting on a random feature at a random threshold
- Path length = number of splits to isolate a single data point
- Short path → point is isolated quickly → anomaly
- Long path → point blends in with many others → normal
- Anomaly score averaged across all trees; scores below contamination threshold labelled as outliers

**[Diagram — typed ASCII or a simple hand-drawn style if you want an image slot]**

*Presenter notes: Use the "person in a forest" analogy. "Normal people need many questions to identify. Anomalies are obvious." This is memorable.*

---

## Slide 7: Procedure

**[Title]** Analysis Procedure

**[Body — dot points]**
- Step 1: Load and inspect data; check shape, missing values, distributions
- Step 2: Filter players with <500 minutes; merge traded player records
- Step 3: Engineer features: salary-per-WS, points-per-dollar, PER-salary-ratio
- Step 4: Log-transform salary; standardise all features (StandardScaler)
- Step 5: Train Isolation Forest (n_estimators=200, contamination=0.10, random_state=42)
- Step 6: Sensitivity test — repeat with contamination = 0.05, 0.15, 0.20
- Step 7: Train Local Outlier Factor (n_neighbors=20) for comparison
- Step 8: Visualise anomaly scores; label flagged players on scatter plot

*Presenter notes: Walk through this briskly — 40 seconds. It shows systematic thinking.*

---

## Slide 8: Results — Scatter Plot

**[Title]** Results: Flagged Salary Anomalies

**[Insert chart]** `outputs/final_result_plot.png`
- Red dots = flagged outliers (salary anomalous vs performance)
- Blue dots = normal (salary consistent with performance)
- Player names labelled on top anomalies

*Presenter notes: This is your MONEY slide. Point to specific names. "The algorithm flagged [name] — earning $X million, with a Win Shares of only Y. That's $Z million per win contributed. Compare that to [underpaid player], earning $A million with Win Shares of B." Then mention LeBron: "LeBron James appears as #5 overpaid — but any basketball fan knows this is debatable. He missed regular season games with injury, lowering his Win Shares, then led the Lakers from a 2-10 start to the Western Conference Finals. This is where human judgment must complement the algorithm — and we'll discuss this more on the next slide." Let the audience read the chart.*

---

## Slide 9: Discussion and Conclusion

**[Title]** Discussion and Conclusion

**[🧠 Domain Knowledge Check: LeBron James — dot points]**
- Algorithm flagged LeBron as #5 overpaid: $44.5M salary, 5.6 WS → $6.7M per win share
- But context matters: LeBron missed 27 regular season games with injury → lower WS accumulation
- He then led the Lakers from a 2-10 start to the Western Conference Finals
- Playoff impact, leadership, and marketability are invisible to the model
- Lesson: **data-driven does not mean data-only** — human judgment must complement algorithmic output

**[Discussion — dot points]**
- Isolation Forest flagged 33 outliers (10.2% of filtered dataset)
- Overpaid outliers share common patterns: **injured veterans** (John Wall — barely played), **aging stars on declining max contracts** (Westbrook), and **"potential tax"** — teams paying for what a player *might* become, not current production (Ben Simmons)
- Top flagged underpaid players are on **rookie-scale contracts** (Desmond Bane, Walker Kessler, Tyrese Maxey) — not "underpaid" by market failure, but by CBA rules that cap early-career earnings
- LOF agreed on 19 of 33 flagged players — consensus cases are strongest candidates for true anomalies
- Isolation Forest detected global outliers (extreme salaries) that LOF missed due to its local density approach — IF is better suited for this problem

**[Limitations — dot points]**
- Single-season snapshot (2022-23 only) — does not reflect career trajectory; a player may look overpaid this year but was fairly compensated across a multi-year deal
- Salary reflects expected future value, not just current performance — a rookie on a cheap deal isn't "underpaid" long-term
- Intangible value is invisible: a player's cultural influence (LeBron, Curry), jersey sales, city pride, TV ratings, and brand value to the NBA cannot be captured by box-score stats
- Win Shares is a counting stat — injured players accumulate fewer WS even if per-game impact is elite (the LeBron case)
- Context matters: contracts are signed years in advance based on projected performance, not current-season stats alone

**[Conclusion — dot points]**
- Isolation Forest provides an objective, scalable framework for salary anomaly detection
- Combining with LOF and domain knowledge creates a more robust analysis than any single method alone
- Future work: multi-season data, SHAP feature importance, positional stratification

*Presenter notes: Start with the LeBron story — it hooks the audience and shows you're not blindly trusting the algorithm. Then discuss the IF vs LOF comparison — stress that IF is better for this problem because NBA salary outliers are GLOBAL (extreme salaries far from the median), which is exactly what IF is designed for. The limitations show intellectual honesty — this is what the 10% insight mark rewards.*

---

## Slide 10: References

**[Title]** References

**[Body — typed]**
- jamiewelsh2 (2023). NBA Player Salaries (2022–23 Season). Kaggle. https://www.kaggle.com/datasets/jamiewelsh2/nba-player-salaries-2022-23-season
- Liu, F.T., Ting, K.M., & Zhou, Z.H. (2008). Isolation Forest. *2008 Eighth IEEE International Conference on Data Mining*, 413–422.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Basketball Reference (2023). 2022–23 NBA Player Stats. https://www.basketball-reference.com

*Presenter notes: Don't read these out. Just say "references are listed here for full attribution." Move on.*

---

## Timing Guide

| Slide | Target time | Cumulative |
|---|---|---|
| 1 — Title | 0:15 | 0:15 |
| 2 — Declaration | 0:20 | 0:35 |
| 3 — Problem | 1:20 | 1:55 |
| 4 — DM Problem Type | 1:10 | 3:05 |
| 5 — Dataset | 1:15 | 4:20 |
| 6 — Method | 1:30 | 5:50 |
| 7 — Procedure | 0:50 | 6:40 |
| 8 — Results | 1:30 | 8:10 |
| 9 — Discussion | 1:30 | 9:40 |
| 10 — References | 0:15 | 9:55 |

**Target: 9:55 total. Gives you 5 seconds buffer before the 10-minute mark.**
