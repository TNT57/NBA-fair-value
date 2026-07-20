# steps.md — Day-by-Day Execution Plan

> Treat each checkbox as a real task. Don't move to the next day until the current one is done.

---

## Day 1 — Setup + Data + Understanding

### Morning: Environment Setup
- [ ] Install Claude Code extension in VSCode
- [ ] Create project folder: `nba-salary-outlier/`
- [ ] Install dependencies (see `requirements.txt` below)
- [ ] Create subfolder structure:
  ```
  nba-salary-outlier/
  ├── data/          ← raw CSV files go here
  ├── notebooks/     ← your .ipynb or .py analysis files
  ├── outputs/       ← saved plots and results
  └── slides/        ← your PowerPoint file
  ```

### Afternoon: Get the Data
- [ ] Go to: https://www.kaggle.com/datasets/jamiewelsh2/nba-player-salaries-2022-23-season
- [ ] Download the CSV — save to `data/` folder
- [ ] Also download (optional but helpful for richer features):
  https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats
  (filter to 2022–23 season for per-game stats)
- [ ] Open the CSV and look at it manually — understand what each column means
- [ ] Read `method.md` fully — understand Isolation Forest before coding

### Evening: Exploratory Data Analysis (EDA)
- [ ] Load data in Python, check shape, dtypes, null values
- [ ] Plot salary distribution — confirm it is right-skewed
- [ ] Plot key stats (PTS, WS, PER) distributions
- [ ] Check correlation between salary and performance metrics
- [ ] Save at least 2 plots for your slides (salary distribution, salary vs WS scatter)

**Day 1 done when:** You have clean data loaded, understand its shape, and have 2 saved visualisation plots.

---

## Day 2 — Preprocessing + Modelling

### Morning: Preprocessing
- [ ] Filter out players with <500 minutes played (too small a sample to be meaningful)
- [ ] Handle missing values (drop or impute — justify your choice)
- [ ] Engineer these features:
  - `salary_per_WS` = Salary / (Win Shares + 0.1)  ← add 0.1 to avoid division by zero
  - `pts_per_dollar` = PTS / (Salary / 1_000_000)
  - `PER_salary_ratio` = PER / (Salary / 1_000_000)
- [ ] Log-transform salary: `log_salary = np.log(salary)`
- [ ] Standardise all features with `StandardScaler`
- [ ] Split into feature matrix X (no salary as raw — use log_salary instead)

### Afternoon: Isolation Forest
- [ ] Train Isolation Forest:
  ```python
  from sklearn.ensemble import IsolationForest
  model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
  model.fit(X)
  scores = model.decision_function(X)
  labels = model.predict(X)  # -1 = outlier, 1 = normal
  ```
- [ ] Try at least 2 contamination values (0.05 and 0.15) — compare results
- [ ] Save anomaly scores and labels back to your DataFrame

### Late Afternoon: LOF Comparison
- [ ] Train Local Outlier Factor:
  ```python
  from sklearn.neighbors import LocalOutlierFactor
  lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
  lof_labels = lof.fit_predict(X)
  ```
- [ ] Compare which players each method flags — note agreements and disagreements

### Evening: Visualisation
- [ ] Scatter plot: Win Shares (x) vs Salary (y), coloured by IF label (red=outlier, grey=normal)
- [ ] Scatter plot: PER (x) vs Salary (y), same colouring
- [ ] Label the top 5 most anomalous players on the plot
- [ ] Save all plots to `outputs/`

**Day 2 done when:** You have flagged outliers, comparison between IF and LOF, and 3+ saved plots.

---

## Day 3 — Analysis + Slides

### Morning: Write Your Discussion
Answer these questions in your own words (you'll use this in the presentation):
- [ ] Which players were flagged as overpaid? Does it make sense intuitively?
- [ ] Which players were flagged as underpaid? Does it make sense?
- [ ] Where did IF and LOF disagree? Why might that be?
- [ ] What are 3 real limitations of your approach?
- [ ] What would you do with more time / data?

### Afternoon: Build Slides
- [ ] Open `slides.md` in this folder — it has the exact content for each slide
- [ ] Build in PowerPoint (or Google Slides exported as .pptx)
- [ ] Follow these design rules:
  - Consistent font (e.g. Calibri or Arial, 24–28pt for body)
  - Dark background or clean white — pick one and stick with it
  - Dot points, NOT paragraphs
  - Max 5–6 dot points per slide
  - Images/charts on their own slides or beside minimal text
  - No animations (they waste time and look amateur)

**Day 3 done when:** All 10 slides drafted, plots inserted, no paragraphs of text anywhere.

---

## Day 4 — Polish + First Run-Through

- [ ] Read every slide aloud — fix typos and awkward phrasing
- [ ] Time yourself presenting from slide 1 to slide 9 — aim for 10:00 exactly
- [ ] If under 9:00 — add more detail to Results/Discussion
- [ ] If over 11:00 — cut content from Method or Procedure (most common bloat area)
- [ ] Check: is every claim on a slide something you can explain if asked?
- [ ] Make sure your declaration slide is there

---

## Day 5 — Final Practice

- [ ] Do 2 full run-throughs, timed
- [ ] Practice explaining Isolation Forest without looking at slides — just talking
- [ ] Prepare answers to likely questions (see `presentation.md`)
- [ ] Export/save final .pptx
- [ ] Submit on time

---

## requirements.txt (Python packages)

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
openpyxl
```

Install with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter openpyxl
```
