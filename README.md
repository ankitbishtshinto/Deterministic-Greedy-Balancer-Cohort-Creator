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
