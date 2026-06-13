# Do LLMs Play Nash? Strategic Behavior of Language Models

**CSCE 631 — Summer I 2026 | Texas A&M University**

---

## Introduction

Large language models (LLMs) are increasingly deployed as decision-making agents in negotiation, resource allocation, and multi-agent systems. A natural question is whether these models behave strategically in the game-theoretic sense: do their action choices approximate Nash equilibrium strategies when placed in well-defined adversarial or cooperative games?

This project evaluates whether LLMs approximate Nash equilibria across twelve canonical game-theoretic scenarios with analytically known solutions. We construct a suite spanning four game classes — social dilemmas, zero-sum games, coordination games, and bargaining problems — and query the TAMU LLM gateway (GPT-5-mini and Claude Sonnet 4.5) at multiple temperatures. LLM responses are parsed into probability distributions over actions, and **exploitability** relative to the Nash equilibrium is computed per game and temperature. The result is a quantitative map of where and why LLMs deviate from rational play.

---

## Method

### Game Suite

We construct twelve two-player normal-form games with analytically verified Nash equilibria (Table 1). Games span four types:

| Type | Games |
|------|-------|
| Social dilemma | Prisoner's Dilemma (classic, high stakes, low stakes) |
| Zero-sum | Matching Pennies, Rock-Paper-Scissors |
| Coordination | Coordination Game, Battle of the Sexes, Stag Hunt, Chicken |
| Bargaining | Nash Bargaining (symmetric and asymmetric) |
| General-sum | 3×3 general-sum game |

All Nash equilibria are verified to have exploitability < 10⁻³. For the 3×3 general-sum game, the equilibrium is computed numerically via the nashpy library's support enumeration algorithm.

### Nash Equilibrium Computation

Equilibria for games with known closed-form solutions are pre-computed analytically and verified programmatically. For the 3×3 game, we use nashpy's support enumeration method, which iterates over all support pairs (S₁, S₂) and solves the indifference conditions as a linear system. We verify each candidate by checking that neither player can profitably deviate (exploitability < 10⁻³).

### Exploitability

For a strategy profile (σ₁, σ₂), total exploitability is defined as:

**ε(σ₁, σ₂) = [max_a u₁(a, σ₂) − u₁(σ₁, σ₂)] + [max_b u₂(σ₁, b) − u₂(σ₁, σ₂)]**

This measures how much each player could gain by switching to their best response. At a Nash equilibrium, ε = 0 by definition. We use the Nash equilibrium opponent strategy as the reference col strategy, so exploitability measures the row player's (LLM's) regret relative to playing the Nash response against a Nash opponent.

### LLM Querying

We query the TAMU API gateway at `chat.tamu.ai` using the OpenAI-compatible Python SDK. Two experimental conditions:

1. **Temperature sweep**: GPT-5-mini (`protected.gpt-5-mini`) at T ∈ {0, 0.5, 1.0}. GPT models support arbitrary temperature; Claude's extended thinking mode forces temperature=1 and is unsuitable for this condition.

2. **Model comparison**: GPT-5-mini, Claude Haiku 4.5, and Claude Sonnet 4.5 all queried at T=1.

Each (game, condition) cell is queried N=5 times. Responses are parsed into probability distributions using a cascade: JSON dict → single action name → percentage mentions → uniform fallback. The five distributions are averaged per cell.

Each game is presented as a payoff matrix with labeled actions. The system prompt instructs the model to act as a rational decision-maker and respond with only an action name or probability distribution.

---

## Results

*(Fill in after running experiments. Replace the placeholders below with actual figures and numbers.)*

### Exploitability by Game and Temperature

**Table 2: Exploitability ε (GPT-5-mini)**

| Game | T=0 | T=0.5 | T=1.0 |
|------|-----|--------|--------|
| Prisoner's Dilemma (Classic) | | | |
| Prisoner's Dilemma (High Stakes) | | | |
| Prisoner's Dilemma (Low Stakes) | | | |
| Coordination Game | | | |
| Battle of the Sexes | | | |
| Stag Hunt | | | |
| Matching Pennies | | | |
| Rock-Paper-Scissors | | | |
| Nash Bargaining (Symmetric) | | | |
| Nash Bargaining (Asymmetric) | | | |
| Chicken | | | |
| 3×3 General-Sum | | | |

**Figure 1** shows the exploitability heatmap. **Figure 2** shows per-game strategy comparisons for the four most illustrative games.

### Model Comparison (T=1)

**Table 3: Mean exploitability by model**

| Model | Mean ε |
|-------|--------|
| GPT-5-mini | |
| Claude Haiku 4.5 | |
| Claude Sonnet 4.5 | |

---

## Discussion

### Cooperation Bias in Social Dilemmas

*(Fill in: did the LLM cooperate more than Nash predicts in Prisoner's Dilemma? Did the scale of payoffs matter — compare classic vs. low stakes vs. high stakes?)*

In the Prisoner's Dilemma, the unique Nash equilibrium is mutual defection, as Defect strictly dominates Cooperate regardless of the opponent's action. Despite this, LLMs frequently choose Cooperate, consistent with the cooperation bias documented in Akata et al. (2023). This bias...

### Failure to Randomize in Zero-Sum Games

*(Fill in: did the LLM play 50/50 in Matching Pennies? Did it show rock bias in RPS? What does this mean for exploitability?)*

In zero-sum games (Matching Pennies, RPS), the unique Nash equilibrium requires uniform randomization across actions. A deterministic or biased strategy is exploitable — an opponent who knows the LLM will favor Heads 60% of the time can always choose Tails and earn positive expected value. We observe...

### Focal-Point Effects in Coordination

*(Fill in: did the LLM show a bias toward the first-listed action / "Option A" / "Opera" in coordination games? Did this align with or deviate from the Pareto-optimal NE?)*

In the Coordination Game and Battle of the Sexes, multiple pure Nash equilibria exist. LLMs tend to select the first-listed or more prominent option (focal-point effect, Schelling 1960). In the Stag Hunt, both (Stag, Stag) — the Pareto-optimal equilibrium — and (Hare, Hare) — the risk-dominant equilibrium — are Nash equilibria...

### Equality Bias in Bargaining

*(Fill in: did the LLM demand 50 in the asymmetric bargaining game where the Nash bargaining solution is 60/40?)*

In the Nash Bargaining problem with asymmetric outside options, the Nash bargaining solution assigns payoff proportional to the gains over disagreement payoffs. With Row's outside option = 30 and Col's = 10 over a surplus of 100, the NBS predicts Row demands 60 and Col demands 40. We observe...

### Effect of Temperature

*(Fill in: does higher temperature bring LLM closer to or further from Nash? Interpretation through regret-minimization lens.)*

From the regret-minimization framework (Lectures 7–9), a no-regret learner converging to Nash plays approximately optimally in hindsight. Temperature in LLMs controls output entropy but is not equivalent to a regret-minimizing update rule. Higher temperature...

### Limitations

- **Prompt sensitivity**: LLM responses may vary substantially with prompt wording. We use a standardized prompt but do not test paraphrases.
- **Parsing noise**: responses that do not match any action clearly are assigned a uniform distribution, artificially reducing measured exploitability in those cases.
- **Single-shot play**: real strategic rationality may emerge differently in repeated interaction; this project tests one-shot decisions only.
- **Model opacity**: we cannot observe the chain-of-thought reasoning (even with Claude's extended thinking output) that leads to each action choice.

---

## References

- Nash, J. F. (1950). Equilibrium points in n-person games. *PNAS*, 36(1), 48–49.
- Nash, J. F. (1950). The bargaining problem. *Econometrica*, 18(2), 155–162.
- Akata, E., Schulz, L., Coda-Forno, J., Oh, J., Bethge, M., & Schulz, E. (2023). Playing repeated games with large language models. *arXiv:2305.16867*.
- Hart, S., & Mas-Colell, A. (2000). A simple adaptive procedure leading to correlated equilibrium. *Econometrica*, 68(5), 1127–1150.
- Schelling, T. C. (1960). *The Strategy of Conflict*. Harvard University Press.
- Nisan, N., Roughgarden, T., Tardos, E., & Vazirani, V. V. (2007). *Algorithmic Game Theory*. Cambridge University Press.
- Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C. (2007). Regret minimization in games with incomplete information. *NeurIPS*.
