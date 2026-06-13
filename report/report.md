# Do LLMs Play Nash? Strategic Behavior of Language Models
**CSCE 631 — Summer I 2026 | Texas A&M University**

---

## Introduction

<!-- 
Problem statement and motivation.
- Do LLMs approximate Nash equilibria in canonical game-theoretic scenarios?
- Motivate the question: LLMs are used as decision-making agents; understanding their strategic rationality matters.
-->

## Method

<!--
- Game suite description (12 games, 3 types)
- Nash computation method (support enumeration + LP via nashpy)
- LLM querying procedure (TAMU API, temperatures 0 / 0.5 / 1.0, N=5 queries per cell)
- Exploitability formula and how LLM responses are parsed into distributions
-->

## Results

<!--
- Table: exploitability per game × temperature
- Heatmap figure
- Per-game strategy comparison bar charts (select 3–4 most interesting games)
- Cooperation bias summary figure
-->

## Discussion

<!--
- Connect findings to game-theoretic concepts:
  * Cooperation bias in social dilemmas (PD)
  * Focal-point effects in coordination games
  * Failure to randomize in zero-sum games (Matching Pennies, RPS)
  * Equality bias in bargaining vs. NBS
- Role of temperature on exploitability
- Limitations: single model, prompt sensitivity, parsing noise
- Extensions: multi-round play, different models, chain-of-thought prompting
-->

## References

- Nash, J. F. (1950). Equilibrium points in n-person games. *PNAS*.
- Nash, J. F. (1950). The bargaining problem. *Econometrica*.
- Akata, E., Schulz, L., Coda-Forno, J., Oh, J., Bethge, M., & Schulz, E. (2023). Playing repeated games with large language models. *arXiv:2305.16867*.
- Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C. (2007). Regret minimization in games with incomplete information. *NeurIPS*.
- Nisan, N., Roughgarden, T., Tardos, E., & Vazirani, V. V. (2007). *Algorithmic Game Theory*. Cambridge University Press.
