"""
Run all LLM queries and save results to data/responses/.
Run from repo root: python scripts/run_queries.py
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import json
import time
from pathlib import Path
from tqdm import tqdm

from src.games import build_game_suite
from src.llm import TAMUClient
from src.llm.response_parser import build_game_prompt

RESPONSES_DIR = Path('data/responses')
RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

TEMPERATURES = [0.0, 0.5, 1.0]
N_QUERIES = 5

games = build_game_suite()
print(f"Loaded {len(games)} games\n")

# ── Temperature sweep (GPT-mini) ─────────────────────────────────────────────
sweep_path = RESPONSES_DIR / 'responses_temp_sweep.json'
temp_cache = json.loads(sweep_path.read_text()) if sweep_path.exists() else {}

print("=== Temperature sweep: gpt-5-mini at T=0, 0.5, 1.0 ===")
client = TAMUClient(model='gpt-mini', temperature=1.0)

for g in tqdm(games, desc='Games'):
    if g.name not in temp_cache:
        temp_cache[g.name] = {}

    prompt = build_game_prompt(
        g.name, g.row_actions, g.col_actions,
        g.row_payoffs, g.col_payoffs, player=1
    )

    for T in TEMPERATURES:
        key = str(T)
        existing = temp_cache[g.name].get(key, [])
        needed = N_QUERIES - len(existing)
        if needed <= 0:
            tqdm.write(f"  [cached] {g.name[:40]} T={T}")
            continue

        responses = list(existing)
        for n in range(needed):
            resp = client.query_game(prompt, temperature=T)
            responses.append(resp)
            tqdm.write(f"  {g.name[:35]:35s} T={T} #{len(responses)}: {resp[:60]}")
            time.sleep(1)
        temp_cache[g.name][key] = responses

sweep_path.write_text(json.dumps(temp_cache, indent=2))
print(f"\nTemperature sweep saved → {sweep_path}")
print(f"Total tokens used: {client.total_tokens_used:,}\n")

# ── Model comparison (GPT-mini, Haiku, Sonnet at T=1) ────────────────────────
model_path = RESPONSES_DIR / 'responses_model_compare.json'
model_cache = json.loads(model_path.read_text()) if model_path.exists() else {}

COMPARE_MODELS = {
    'gpt-mini': 'gpt-mini',
    'haiku':    'haiku',
    'sonnet':   'sonnet',
}

print("=== Model comparison at T=1 ===")
for model_key in COMPARE_MODELS:
    client = TAMUClient(model=model_key, temperature=1.0)
    print(f"\nQuerying {client.model_id}...")

    for g in tqdm(games, desc=model_key, leave=False):
        cache_key = f'{model_key}|{g.name}'
        existing = model_cache.get(cache_key, [])
        needed = N_QUERIES - len(existing)
        if needed <= 0:
            continue

        prompt = build_game_prompt(
            g.name, g.row_actions, g.col_actions,
            g.row_payoffs, g.col_payoffs, player=1
        )
        responses = list(existing)
        for _ in range(needed):
            resp = client.query_game(prompt)
            responses.append(resp)
            tqdm.write(f"  {model_key:10s} {g.name[:35]:35s}: {resp[:60]}")
            time.sleep(1)
        model_cache[cache_key] = responses

    model_path.write_text(json.dumps(model_cache, indent=2))
    print(f"  Tokens: {client.total_tokens_used:,}")

print(f"\nModel comparison saved → {model_path}")
print("\nDone. Open the notebook and run cells 5-7 to see results.")
