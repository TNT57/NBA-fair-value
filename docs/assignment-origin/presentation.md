# presentation.md — How to Present This Well

---

## The Golden Rule

Your slides are speaker notes for your AUDIENCE.
Your brain is the speaker notes for YOU.
Never read from the slides — talk to the room.

---

## Opening Hook (memorise this — don't read it)

> "In 2023, Ben Simmons was paid $37 million dollars by the Brooklyn Nets.
> He played 42 games. That works out to roughly $880,000 per game —
> for a player who, by most efficiency metrics, was performing at a
> role-player level. Meanwhile, players earning a fraction of that salary
> were contributing more wins to their teams. How does this happen?
> And more importantly — can we build a system that catches it?
> That's exactly what this project does."

This takes about 20 seconds. It's conversational. It's surprising. It works.

---

## Slide-by-Slide Speaking Notes

### Slide 3 (Problem) — Key things to say out loud, not on slides
- "The NBA salary cap is a zero-sum game. If you overpay one player, you have less to build around them."
- "Traditional valuation is based on reputation — who won a championship three years ago, who has the highlight reel. That's not data."
- "We're asking: can an algorithm look at a player's actual on-court numbers and tell us if their pay cheque is justified?"

### Slide 4 (DM Problem Type) — Be ready to justify this
- "This is NOT a classification problem — we don't have labelled examples of overpaid players. No one has certified ground truth."
- "This is NOT regression — we're not predicting what a player should earn. We're detecting who deviates from the norm."
- "Outlier detection is the right tool because we want to find the exceptions — the players who don't fit the pattern everyone else follows."

### Slide 6 (Method) — The forest analogy
- "Think of it like this. Imagine you're in a room of 350 people and you're trying to find the odd one out. A truly normal person blends in — you'd need to ask them many questions to distinguish them. But someone who is very unusual gets isolated very quickly. Isolation Forest works the same way — it uses random splitting to see who gets isolated fast."
- Don't rush this. It's the most important conceptual slide.

### Slide 8 (Results) — Drive the narrative
- Point at specific red dots. Name them if you know who they are from your results.
- "This player here — [name] — was earning $X million with a Win Share of Y. That puts their cost-per-win at $Z million. The algorithm flagged them as the most anomalous player in the dataset."
- "Down here in the green zone — players with high win shares on modest contracts. These are the undervalued targets that smart front offices look for."
- Pause here. Let the audience look at the chart. It's your strongest visual.

### Slide 9 (Discussion) — Show depth
- "I want to be honest about what this model can and can't do."
- "Salary isn't just about last season's numbers. Contracts are signed based on expected future performance. A player coming off an injury might look overpaid on last year's stats but is being paid for what they're projected to do next year."
- "That said, the model's flagged cases largely align with what NBA analysts and journalists have been saying. That gives us some confidence in the approach."

---

## Q&A Prep — Questions Your Professor Might Ask

**Q: Why did you choose contamination = 0.10?**
A: "I chose 0.10 based on domain reasoning — roughly 10% of NBA players at any time are considered significantly misvalued by analysts and sports economists. I also ran sensitivity tests at 0.05 and 0.15, and the core results were consistent. The players flagged in all three settings represent the strongest cases."

**Q: How do you know these players are actually overpaid and not just injured?**
A: "That's exactly the right question, and it's one of the key limitations. The model is working with one season of data and doesn't know the injury context. What it tells us is statistical anomaly — not causation. A follow-up study would add injury history as a feature. That said, several flagged players do align with publicly known cases of injury-hampered seasons."

**Q: Why Isolation Forest over One-Class SVM?**
A: "One-Class SVM requires careful kernel selection and is sensitive to hyperparameters. It's also computationally slower. Isolation Forest is specifically designed for tabular data like this, runs in linear time, and doesn't assume a normal distribution — which matters because our salary distribution is right-skewed. It's also been shown empirically to outperform distance-based methods on datasets of this size."

**Q: What does the anomaly score actually mean?**
A: "The anomaly score from Isolation Forest's decision_function method represents the average normalised path length across all trees. More negative values mean the point was isolated more quickly — it's the most deviant from the rest of the data. A score around zero is borderline, and positive scores are clearly normal. I ranked players by this score to find the most anomalous cases."

**Q: Could you use this to build a better team?**
A: "In theory, yes — and that's the real-world application. A front office could use this to identify free agents who are underpaid relative to their statistical contribution and target them before their value is widely recognised. It's the core idea behind Moneyball-style analytics, now applied to salary structure rather than draft picks."

**Q: Why did you filter out players with less than 500 minutes?**
A: "Players with very few minutes have extremely noisy per-game statistics. A player who plays 50 minutes in one game and scores 30 points looks like a superstar statistically, but that's not a real signal. 500 minutes — roughly 10 games of full playing time — gives a more stable sample. This is a common threshold in NBA analytics research."

---

## Timing Practice Strategy

1. First run-through: don't worry about time, just talk naturally through all slides
2. Second run-through: time it. Note which sections ran long.
3. Third run-through: cut ruthlessly from the Method slide if you're over 11 minutes
4. Fourth run-through: aim for 9:40–9:55. This is your target.

Common causes of going under 9 minutes:
- Rushing through Results (don't — this is the payoff slide, slow down)
- Skipping the forest analogy (don't — it's memorable and earns marks)
- Not naming actual players in your results (don't — it makes the results concrete)

Common causes of going over 11 minutes:
- Too much detail on the algorithm internals
- Reading slide text instead of talking over it
- Pausing too long at each bullet point

---

## Body Language Checklist

- [ ] Make eye contact with different parts of the room — not just the lecturer
- [ ] Point at your slides when referencing a specific chart or dot point
- [ ] Slow down when you say something important — silence creates emphasis
- [ ] Don't hold a pen/clicker nervously — set it down between slides
- [ ] Breathe before the Results slide — it's the most important one, don't rush
