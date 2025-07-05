# Deterministic-Greedy-Balancer-Cohort-Creator
Creates balanced cohorts

A simple Python module that:
- **Balances head-count** exactly according to your desired percentages.  
- **Creates cohorts** with balanced key metrics (searches, bookings, etc.) using a deterministic greedy algorithm.

---

## Features

- **Flexible cohorts**: Define any number of groups by name, each with its own target percentage.  
- **Custom metrics**: Balance on whichever numeric columns you choose (e.g., searches, bookings, cancellations, GTV).  
- **Error margin**: Optionally allow small deviations in group sizes (default 1%).  
- **Deterministic**: Same inputs always yield the same assignments.

---

## Installation

1. Ensure Python 3.7+ is installed.  
2. Install required libraries:
   ```bash
   pip install pandas numpy
   ```
3. Copy `deterministic_greedy_balancer.py` (or your chosen filename) into your project.

---
## Usage

```python
from deterministic_greedy_balancer import build_cohort_comb_ind

# 1. Prepare a DataFrame (must include an 'experiment_type' column):
#    df = pd.read_csv('my_data.csv')

# 2. Define your cohorts:
user_cohorts = [
    {'cohort_name': 'Control',  'split_percentage': 0.40},
    {'cohort_name': 'VariantA', 'split_percentage': 0.30},
    {'cohort_name': 'VariantB', 'split_percentage': 0.30},
]

# 3. List the metric columns to balance on:
metrics = ['searches', 'bookings', 'cancellations', 'gtv']

# 4. Run the cohort creator:
cohort_df = build_cohort_comb_ind(
    df,
    cohorts=user_cohorts,
    metric_columns=metrics,
    error_margin=0.01  # optional, default is 0.01
)

# 5. Inspect results:
print(cohort_df.head())
```

The result is a DataFrame with all original columns plus a new `group` column indicating cohort.

---

## How It Works (Simple Explanation)

1. **Normalize metrics** to 0–1 so they’re comparable.  
2. **Compute a composite score** by averaging (or weighting) those normalized metrics.  
3. **Sort** items by composite score from highest to lowest.  
4. **Calculate** exact item counts per cohort (e.g., 40%, 30%, 30%).  
5. **Greedy assignment**: For each item in rank order, try placing it in each cohort, compute which choice keeps each group’s metric shares closest to the target, then assign it where the “cost” is smallest.  
6. Repeat until all items are assigned.

---

## Tips

- Supports **any number** of cohorts and splits (not just two or three).  
- Adjust `error_margin` for stricter or looser head-count enforcement.  
- You can also pass **weights** into `DeterministicGreedyBalancer` if you want to emphasize certain metrics.

---

---