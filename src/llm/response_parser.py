"""
Parse LLM text responses into probability distributions over game actions.

Strategy:
1. Try to extract a JSON dict mapping action -> probability.
2. Try to find an action name mentioned directly (pure strategy).
3. Fall back to uniform distribution with a warning.
"""
import re
import json
import numpy as np
from typing import List


def parse_action_distribution(response: str, actions: List[str]) -> np.ndarray:
    """
    Convert an LLM response string into a probability distribution over `actions`.
    Returns a numpy array of shape (len(actions),) summing to 1.
    """
    response_clean = response.strip()

    # Attempt 1: JSON dict in the response
    dist = _try_parse_json(response_clean, actions)
    if dist is not None:
        return dist

    # Attempt 2: single action name mentioned
    dist = _try_pure_strategy(response_clean, actions)
    if dist is not None:
        return dist

    # Attempt 3: percentage mentions ("50% Heads, 50% Tails")
    dist = _try_percentage_mix(response_clean, actions)
    if dist is not None:
        return dist

    # Fallback: uniform
    print(f"[WARN] Could not parse response, using uniform distribution.\n  Response: {response_clean[:120]}")
    return np.ones(len(actions)) / len(actions)


def _try_parse_json(response: str, actions: List[str]) -> np.ndarray | None:
    match = re.search(r'\{[^}]+\}', response)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
        probs = np.zeros(len(actions))
        action_lower = [a.lower() for a in actions]
        for key, val in obj.items():
            key_l = key.lower().strip()
            for i, a in enumerate(action_lower):
                if key_l in a or a in key_l:
                    probs[i] = float(val)
                    break
        if probs.sum() > 0:
            return probs / probs.sum()
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _try_pure_strategy(response: str, actions: List[str]) -> np.ndarray | None:
    response_l = response.lower()
    matched = []
    for i, action in enumerate(actions):
        if action.lower() in response_l:
            matched.append(i)
    if len(matched) == 1:
        probs = np.zeros(len(actions))
        probs[matched[0]] = 1.0
        return probs
    # If multiple actions mentioned, can't determine pure strategy
    return None


def _try_percentage_mix(response: str, actions: List[str]) -> np.ndarray | None:
    probs = np.zeros(len(actions))
    found_any = False
    response_l = response.lower()
    for i, action in enumerate(actions):
        action_l = re.escape(action.lower())
        # "Cooperate 40%" or "cooperate: 40%" — action then nearby number%
        # "40% Cooperate" or "40% on Cooperate" — number% then nearby action
        # Use [^%]* to avoid crossing other percentage values
        patterns = [
            rf'{action_l}[^%\d]{{0,20}}(\d+(?:\.\d+)?)\s*%',
            rf'(\d+(?:\.\d+)?)\s*%[^%\d]{{0,20}}{action_l}',
        ]
        for pat in patterns:
            m = re.search(pat, response_l)
            if m:
                probs[i] = float(m.group(1)) / 100.0
                found_any = True
                break
    if found_any and probs.sum() > 0:
        return probs / probs.sum()
    return None


def build_game_prompt(game_name: str, row_actions: List[str], col_actions: List[str],
                      row_payoffs, col_payoffs, player: int = 1) -> str:
    """Generate a standardized LLM prompt for a normal-form game."""
    action_list = row_actions if player == 1 else col_actions
    opp_list = col_actions if player == 1 else row_actions

    payoff_desc = _format_payoff_matrix(row_actions, col_actions, row_payoffs, col_payoffs)

    prompt = f"""You are Player {player} in the game "{game_name}".

Your available actions: {action_list}
Your opponent's available actions: {opp_list}

Payoff matrix (your payoff, opponent's payoff):
{payoff_desc}

What action do you choose? Respond with ONLY the action name, or if you want to play a mixed strategy, respond with a JSON object like {{"Action1": 0.5, "Action2": 0.5}}.
Your choice:"""
    return prompt


def _format_payoff_matrix(row_actions, col_actions, row_payoffs, col_payoffs) -> str:
    header = "         " + "  ".join(f"{c:>12}" for c in col_actions)
    rows = [header]
    for i, r in enumerate(row_actions):
        cells = "  ".join(f"({row_payoffs[i,j]:.0f}, {col_payoffs[i,j]:.0f})".rjust(12)
                          for j in range(len(col_actions)))
        rows.append(f"{r:>8}: {cells}")
    return "\n".join(rows)
