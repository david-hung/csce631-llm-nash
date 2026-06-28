# Do LLMs Play Nash? Strategic Behavior of Language Models
**CSCE 631 — Summer I 2026 | Texas A&M University**

Final submission due: **June 29, 2026**

## Project Summary

This project evaluates whether large language models (LLMs) approximate Nash equilibrium strategies when presented with canonical game-theoretic scenarios. A suite of 10+ normal-form and bargaining games with analytically known equilibria is constructed. Each game is queried via the TAMU API at temperatures T = 0, 0.5, and 1.0. LLM responses are converted to probability distributions over actions, and exploitability relative to the Nash equilibrium is computed per game and temperature.

Key questions:
- Do LLMs approximate Nash equilibria in zero-sum games (e.g., Matching Pennies)?
- Do they exhibit cooperation bias in social dilemmas (e.g., Prisoner's Dilemma)?
- Do focal-point effects or fairness norms distort bargaining solutions?

## Repository Structure

```
├── notebooks/
│   └── llm_nash_analysis.ipynb   # Main Jupyter notebook (primary deliverable)
├── src/
│   ├── games/                    # Game payoff matrices + ground-truth NE
│   ├── solvers/                  # Nash solver + exploitability
│   ├── llm/                      # TAMU API client + response parser
│   └── analysis/                 # Plots and aggregation
├── data/
│   └── responses/                # Cached LLM response JSON
├── report/
│   └── report.tex                # LaTeX report source
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

### TAMU API
Log into [chat.tamu.ai](https://chat.tamu.ai) with your NetID + Duo MFA, then copy the `CF_Authorization` cookie from DevTools (Application → Cookies). Add to a `.env` file:
```
TAMU_CF_COOKIE=CF_Authorization=eyJ...
TAMU_API_KEY=<your-api-key>
```

## Running the Notebook

```bash
jupyter notebook notebooks/llm_nash_analysis.ipynb
```

## Game Suite

| # | Game | NE Type |
|---|------|---------|
| 1 | Prisoner's Dilemma (classic) | Pure strategy |
| 2 | Prisoner's Dilemma (high stakes) | Pure strategy |
| 3 | Prisoner's Dilemma (low stakes) | Pure strategy |
| 4 | Coordination Game | Multiple pure NE |
| 5 | Battle of the Sexes | Mixed + 2 pure NE |
| 6 | Stag Hunt | 2 pure NE (risk-dominant vs Pareto) |
| 7 | Matching Pennies | Mixed-strategy NE |
| 8 | Rock-Paper-Scissors | Mixed-strategy NE |
| 9 | Nash Bargaining (split 100) | 50/50 split |
| 10 | Nash Bargaining (asymmetric) | Weighted split |
| 11 | Chicken / Hawk-Dove | Mixed + 2 pure NE |
| 12 | 3×3 General-Sum Game | Mixed-strategy NE |

## Grading Rubric (50% of final grade)

| Component | Weight |
|-----------|--------|
| Proposal | 5% |
| Technical execution | 20% |
| Analysis and insight | 15% |
| Writing quality | 10% |
