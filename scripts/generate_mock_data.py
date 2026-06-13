"""
Generate mock LLM responses for all games and temperatures.
Lets you run the full notebook pipeline before you have the TAMU cookie.

Mock responses simulate plausible LLM biases:
  - Social dilemmas: cooperation bias (LLM cooperates more than Nash predicts)
  - Zero-sum games: rock/heads bias (LLM under-randomizes)
  - Coordination: focal-point bias toward first option
  - Bargaining: equality bias (always demands 50)

Run from repo root:
  python scripts/generate_mock_data.py
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np
from pathlib import Path
from src.games import build_game_suite

RESPONSES_DIR = Path('data/responses')
RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

TEMPERATURES = [0.0, 0.5, 1.0]
N_QUERIES = 5
rng = np.random.default_rng(42)


def mock_response_for_game(game_name: str, actions: list, temperature: float) -> str:
    """
    Return a plausible (biased) mock LLM response string.
    Biases intentionally deviate from Nash to give interesting results.
    """
    name = game_name.lower()

    # Social dilemmas: cooperate with prob ~0.6 regardless of NE (Defect)
    if "prisoner" in name:
        p_cooperate = 0.55 + temperature * 0.1 + rng.normal(0, 0.05)
        p_cooperate = float(np.clip(p_cooperate, 0.1, 0.9))
        if rng.random() < p_cooperate:
            return actions[0]  # Cooperate
        return actions[1]  # Defect

    # Matching pennies: heads bias ~0.6 (should be 0.5)
    if "matching pennies" in name:
        p_heads = 0.60 - temperature * 0.05 + rng.normal(0, 0.05)
        p_heads = float(np.clip(p_heads, 0.1, 0.9))
        if rng.random() < p_heads:
            return actions[0]  # Heads
        return actions[1]  # Tails

    # RPS: rock bias ~0.45 (should be 1/3 each)
    if "rock" in name:
        probs = np.array([0.45, 0.30, 0.25]) + rng.normal(0, 0.03, 3)
        probs = np.clip(probs, 0.01, 1)
        probs /= probs.sum()
        return json.dumps({a: round(float(p), 3) for a, p in zip(actions, probs)})

    # Battle of the sexes: slight bias toward first option (Opera/focal point)
    if "battle" in name:
        p_first = 0.65 + rng.normal(0, 0.08)
        p_first = float(np.clip(p_first, 0.1, 0.9))
        if rng.random() < p_first:
            return actions[0]
        return actions[1]

    # Stag hunt: Pareto-optimal (Stag) bias at low temp, more Hare at high temp
    if "stag" in name:
        p_stag = 0.70 - temperature * 0.15 + rng.normal(0, 0.07)
        p_stag = float(np.clip(p_stag, 0.1, 0.9))
        if rng.random() < p_stag:
            return actions[0]  # Stag
        return actions[1]  # Hare

    # Coordination: strong focal-point bias toward first option
    if "coordination" in name:
        p_a = 0.80 + rng.normal(0, 0.05)
        p_a = float(np.clip(p_a, 0.5, 0.99))
        if rng.random() < p_a:
            return actions[0]
        return actions[1]

    # Chicken: dove bias (avoid conflict)
    if "chicken" in name or "hawk" in name:
        p_dove = 0.70 + rng.normal(0, 0.06)
        p_dove = float(np.clip(p_dove, 0.3, 0.95))
        if rng.random() < p_dove:
            return actions[1]  # Dove
        return actions[0]  # Hawk

    # Bargaining: equality bias — always demand near 50 regardless of NE
    if "bargaining" in name:
        if "asymmetric" in name:
            # Should demand 60, but equality bias pulls toward 50
            target_action = "Demand 50" if rng.random() < 0.65 else "Demand 60"
        else:
            # Symmetric: NE is 50, so this is correct
            target_action = "Demand 50"
        if target_action in actions:
            return target_action
        # fuzzy match
        for a in actions:
            if "50" in a:
                return a
        return actions[len(actions) // 2]

    # 3x3 general sum: slight bias toward first action
    p_first = 0.45 + rng.normal(0, 0.08)
    p_first = float(np.clip(p_first, 0.1, 0.8))
    remaining = (1.0 - p_first) / (len(actions) - 1)
    probs = [remaining] * len(actions)
    probs[0] = p_first
    return json.dumps({a: round(float(p), 3) for a, p in zip(actions, probs)})


def main():
    games = build_game_suite()
    print(f"Generating mock responses for {len(games)} games × {len(TEMPERATURES)} temps × {N_QUERIES} queries")

    # Temperature sweep cache (GPT-mini simulation)
    temp_cache = {}
    for g in games:
        temp_cache[g.name] = {}
        for T in TEMPERATURES:
            responses = [mock_response_for_game(g.name, g.row_actions, T) for _ in range(N_QUERIES)]
            temp_cache[g.name][str(T)] = responses
            print(f"  {g.name[:40]:40s} T={T}  → {responses[0][:50]}")

    path = RESPONSES_DIR / 'responses_temp_sweep.json'
    with open(path, 'w') as f:
        json.dump(temp_cache, f, indent=2)
    print(f"\nSaved temperature sweep mock data → {path}")

    # Model comparison cache
    model_cache = {}
    for model_key in ['gpt-mini', 'haiku', 'sonnet']:
        for g in games:
            # Sonnet plays slightly closer to Nash; GPT-mini has stronger biases
            bias_scale = {'gpt-mini': 1.0, 'haiku': 0.8, 'sonnet': 0.5}[model_key]
            cache_key = f'{model_key}|{g.name}'
            # Use T=1 but vary bias scale by model
            responses = []
            for _ in range(N_QUERIES):
                r = mock_response_for_game(g.name, g.row_actions, temperature=1.0)
                responses.append(r)
            model_cache[cache_key] = responses

    path2 = RESPONSES_DIR / 'responses_model_compare.json'
    with open(path2, 'w') as f:
        json.dump(model_cache, f, indent=2)
    print(f"Saved model comparison mock data → {path2}")
    print("\nDone. You can now run the full notebook without the TAMU cookie.")
    print("Replace mock data with real API responses once you have the cookie.")


if __name__ == '__main__':
    main()
