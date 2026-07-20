# NBA Salary Outlier Detection — Project Guide
## Data Mining Assignment (30%) | Due: 25 May 2026

> Work through these files **in order**. Each one feeds into the next.

---

## File Index

| File | Purpose | When to use |
|---|---|---|
| `README.md` | This file — start here | Now |
| `requirements.md` | Assignment rubric decoded + what you must hit | Read before anything else |
| `steps.md` | Day-by-day execution plan | Your daily checklist |
| `data.md` | Dataset guide — what to download, what to expect | Day 1 |
| `method.md` | How Isolation Forest works — understand before you code | Day 1–2 |
| `code_guide.md` | Full Python code walkthrough with explanations | Day 2 |
| `slides.md` | Slide-by-slide script and dot point content | Day 3 |
| `presentation.md` | Timing guide, what to say, how to handle Q&A | Day 4–5 |

---

## The Project in One Paragraph

You are building a data-driven system that uses **Isolation Forest** (an anomaly/outlier detection algorithm) to identify NBA players whose salaries are statistically anomalous relative to their on-court performance. The real-world problem: NBA teams operate under a hard salary cap, and a single misallocated max contract ($40–50M/year) can cripple a franchise for years. Traditional valuation relies on reputation and gut feel. This project replaces that with objective, data-driven flagging of overpaid and underpaid players.

---

## Method Summary

- **Problem type:** Outlier / Anomaly Detection
- **Algorithm:** Isolation Forest
- **Comparison algorithm:** Local Outlier Factor (LOF)
- **Dataset:** NBA Player Salaries + Performance Stats (Kaggle)
- **Language:** Python (pandas, sklearn, matplotlib, seaborn)

---

## Quick Reminders

- DO NOT use: Association Rules, Decision Trees, K-Means
- AVOID: Linear Regression, Hierarchical Clustering, Naive Bayes, KNN
- Isolation Forest ✅ — not banned, not covered in course
- Presentation must be **9:00–11:00 minutes** exactly
- Slides must be <20% images — rest must be typed text
