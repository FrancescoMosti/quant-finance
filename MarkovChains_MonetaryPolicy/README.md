# Markov Chains in Monetary Policy Modelling

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FrancescoMosti/timeseries-models/blob/main/MarkovChains_MonetaryPolicy/MyMC_MonetaryPolicy.ipynb)

This project applies Markov Chains to replicate and interpret the decision-making process of the U.S. Federal Reserve (FED) using historical macroeconomic data. The goal is to classify and model monetary policy regimes—Dovish, Neutral, Hawkish—based on observable indicators, and to analyze their transitions through a Markov process.

---

## Overview

The analysis is structured in three main experiments, each testing different proxy variables for monetary policy stance:

1. **Fed Funds Rate (FFR)** – Used as a direct proxy for monetary decisions.
2. **Inflation and Unemployment** – Based on economic theory (e.g., the Phillips Curve).
3. **Inflation, Unemployment, and Real GDP Growth** – An extended model adding business cycle information.

For each configuration, we:
- Classify historical periods into monetary regimes.
- Estimate the transition matrix based on empirical state sequences.
- Analyze the stability and realism of the regime-switching process.
- Compare state sequences across models visually and statistically.

---

## File Structure

```
MarkovChains_MonetaryPolicy/
│
├── data/                            # Contains all macroeconomic Excel datasets
│   ├── FEDFUNDS.xlsx
│   ├── UNRATE.xlsx
│   ├── CPIAUCSL_PC1.xlsx
│   └── GDPC1_PC1.xlsx
│
├── MyMC_MonetaryPolicy.ipynb        # Main notebook with full analysis
├── slides_probability.pdf           # Beamer slides presented in class
└── README.md                        # Project documentation
```

---

## How to Run

1. Clone the full repository:
```bash
git clone https://github.com/FrancescoMosti/timeseries-models.git
cd timeseries-models/MarkovChains_MonetaryPolicy
```

2. Ensure you have the required packages:
```bash
pip install pandas numpy matplotlib openpyxl
```

3. Open the notebook:
```bash
jupyter notebook MyMC_MonetaryPolicy.ipynb
```

Alternatively, click the badge at the top of this file to run it directly on Google Colab.

---

## Main Insights

- The Fed Funds Rate alone captures regime shifts but lacks smooth transitions.
- Inflation and Unemployment allow for richer intermediate dynamics, especially in the Neutral state.
- Adding GDP improves the match with observed cycles and provides more realistic persistence in regimes.
- The transition matrices are ergodic, allowing for long-term stationary distribution analysis.
- This framework can be extended toward Hidden Markov Models or Bayesian regime-switching methods.

---

## Limitations

- Assumes time-homogeneous transition probabilities.
- Ignores expectations, forward guidance, and other forward-looking components.
- Not intended as a forecasting or policy tool, but as an exploratory and pedagogical approach.

---

## Author

Francesco Mosti  
Master in Data Science and Statistical Learning (MD2SL)  
University of Florence \& IMT Lucca 
2024/2025 Edition
