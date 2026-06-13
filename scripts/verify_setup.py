"""
Verify the environment and game suite are working correctly (no API needed).
Run this before the notebook:
  python scripts/verify_setup.py
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np

PASS = '  [PASS]'
FAIL = '  [FAIL]'


def check_imports():
    print("Checking imports...")
    try:
        import numpy, scipy, matplotlib, nashpy, tqdm
        print(f"{PASS} numpy {numpy.__version__}, scipy {scipy.__version__}, nashpy {nashpy.__version__}")
    except ImportError as e:
        print(f"{FAIL} Missing package: {e}")
        print("       Run: pip install -r requirements.txt")
        return False
    try:
        import openai
        print(f"{PASS} openai {openai.__version__}")
    except ImportError:
        print(f"{FAIL} openai not installed — run: pip install openai")
        return False
    return True


def check_games():
    print("\nChecking game suite...")
    from src.games import build_game_suite
    games = build_game_suite()
    assert len(games) == 12, f"Expected 12 games, got {len(games)}"
    for g in games:
        assert g.row_payoffs.shape == (g.num_row_actions, g.num_col_actions)
        assert g.col_payoffs.shape == (g.num_row_actions, g.num_col_actions)
    print(f"{PASS} {len(games)} games loaded with correct payoff shapes")
    return games


def check_nash_solver(games):
    print("\nChecking Nash solver...")
    from src.solvers import compute_nash_equilibria, compute_exploitability

    failures = []
    for g in games:
        if g.known_ne:
            ne_list = g.known_ne
        else:
            ne_list = compute_nash_equilibria(g)

        if not ne_list:
            failures.append(f"No NE found for {g.name}")
            continue

        for (r, c) in ne_list[:1]:
            eps = compute_exploitability(g, r, c)
            if eps > 1e-3:
                failures.append(f"{g.name}: NE exploitability={eps:.4f} (expected ~0)")
            else:
                print(f"{PASS} {g.name[:50]:50s} eps={eps:.2e}")

    if failures:
        for f in failures:
            print(f"{FAIL} {f}")
        return False
    return True


def check_response_parser():
    print("\nChecking response parser...")
    from src.llm.response_parser import parse_action_distribution

    actions = ["Cooperate", "Defect"]

    # Pure strategy
    d = parse_action_distribution("Cooperate", actions)
    assert abs(d[0] - 1.0) < 1e-6, f"Expected [1,0], got {d}"

    # JSON mix
    d = parse_action_distribution('{"Cooperate": 0.3, "Defect": 0.7}', actions)
    assert abs(d[0] - 0.3) < 0.01 and abs(d[1] - 0.7) < 0.01, f"JSON parse failed: {d}"

    # Percentage
    d = parse_action_distribution("I'd play Cooperate 40% and Defect 60%", actions)
    assert abs(d[0] - 0.4) < 0.01, f"Percentage parse failed: {d}"

    # Fallback
    d = parse_action_distribution("I cannot decide", actions)
    assert abs(d.sum() - 1.0) < 1e-6

    print(f"{PASS} Response parser handles pure, JSON, percentage, and fallback cases")
    return True


def check_plots():
    print("\nChecking plot utilities...")
    import matplotlib
    matplotlib.use('Agg')
    from src.analysis import plot_exploitability_heatmap, plot_bias_summary

    dummy = {
        "Game A": {0.0: 0.5, 0.5: 0.3, 1.0: 0.1},
        "Game B": {0.0: 0.8, 0.5: 0.6, 1.0: 0.4},
    }
    fig, ax = plot_exploitability_heatmap(dummy)
    import matplotlib.pyplot as plt
    plt.close(fig)

    bias = {"Game A": 0.2, "Game B": -0.1}
    fig, ax = plot_bias_summary(bias)
    plt.close(fig)

    print(f"{PASS} Plot utilities run without error")
    return True


def main():
    print("=" * 60)
    print("CSCE 631 Project — Setup Verification")
    print("=" * 60)

    ok = True
    ok = check_imports() and ok

    if ok:
        games = check_games()
        ok = check_nash_solver(games) and ok
        ok = check_response_parser() and ok
        ok = check_plots() and ok

    print("\n" + "=" * 60)
    if ok:
        print("All checks passed. Run the notebook next:")
        print("  jupyter notebook notebooks/llm_nash_analysis.ipynb")
        print("\nTo test without the TAMU cookie, first generate mock data:")
        print("  python scripts/generate_mock_data.py")
    else:
        print("Some checks failed — fix the issues above before running the notebook.")
    print("=" * 60)


if __name__ == '__main__':
    main()
