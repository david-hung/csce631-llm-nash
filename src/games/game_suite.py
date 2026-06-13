"""
Game suite for Topic 10: Do LLMs Play Nash?

All payoff matrices use the convention: row player is Player 1 (you),
col player is Player 2 (opponent). Each entry is (row_payoff, col_payoff).
Nash equilibria are pre-computed analytically.
"""
import numpy as np
from .normal_form import NormalFormGame


def build_game_suite() -> list[NormalFormGame]:
    games = []

    # ------------------------------------------------------------------
    # 1. Prisoner's Dilemma (classic)
    #    C=Cooperate, D=Defect. Dominant strategy: D.  NE = (D, D)
    # ------------------------------------------------------------------
    games.append(NormalFormGame(
        name="Prisoner's Dilemma (Classic)",
        row_actions=["Cooperate", "Defect"],
        col_actions=["Cooperate", "Defect"],
        row_payoffs=np.array([[3, 0], [5, 1]], dtype=float),
        col_payoffs=np.array([[3, 5], [0, 1]], dtype=float),
        known_ne=[(np.array([0., 1.]), np.array([0., 1.]))],
        description="T=5, R=3, P=1, S=0. Defect is strictly dominant.",
        game_type="social-dilemma",
    ))

    # ------------------------------------------------------------------
    # 2. Prisoner's Dilemma (high stakes)
    # ------------------------------------------------------------------
    games.append(NormalFormGame(
        name="Prisoner's Dilemma (High Stakes)",
        row_actions=["Cooperate", "Defect"],
        col_actions=["Cooperate", "Defect"],
        row_payoffs=np.array([[10, 0], [15, 1]], dtype=float),
        col_payoffs=np.array([[10, 15], [0, 1]], dtype=float),
        known_ne=[(np.array([0., 1.]), np.array([0., 1.]))],
        description="Scaled payoffs. Defect remains strictly dominant.",
        game_type="social-dilemma",
    ))

    # ------------------------------------------------------------------
    # 3. Prisoner's Dilemma (near-symmetric, weak dominance)
    # ------------------------------------------------------------------
    games.append(NormalFormGame(
        name="Prisoner's Dilemma (Low Stakes)",
        row_actions=["Cooperate", "Defect"],
        col_actions=["Cooperate", "Defect"],
        row_payoffs=np.array([[2, 0], [3, 1]], dtype=float),
        col_payoffs=np.array([[2, 3], [0, 1]], dtype=float),
        known_ne=[(np.array([0., 1.]), np.array([0., 1.]))],
        description="Low-magnitude payoffs. Tests whether scale affects cooperation bias.",
        game_type="social-dilemma",
    ))

    # ------------------------------------------------------------------
    # 4. Coordination Game (pure coordination)
    #    Two pure NE: (A,A) and (B,B)
    # ------------------------------------------------------------------
    games.append(NormalFormGame(
        name="Coordination Game",
        row_actions=["Option A", "Option B"],
        col_actions=["Option A", "Option B"],
        row_payoffs=np.array([[1, 0], [0, 1]], dtype=float),
        col_payoffs=np.array([[1, 0], [0, 1]], dtype=float),
        known_ne=[
            (np.array([1., 0.]), np.array([1., 0.])),
            (np.array([0., 1.]), np.array([0., 1.])),
            (np.array([0.5, 0.5]), np.array([0.5, 0.5])),  # mixed NE
        ],
        description="Both players benefit from matching. Tests focal-point bias toward A.",
        game_type="coordination",
    ))

    # ------------------------------------------------------------------
    # 5. Battle of the Sexes
    #    Row prefers (Opera, Opera), Col prefers (Football, Football)
    #    Mixed NE: row plays (2/3, 1/3), col plays (1/3, 2/3)
    # ------------------------------------------------------------------
    games.append(NormalFormGame(
        name="Battle of the Sexes",
        row_actions=["Opera", "Football"],
        col_actions=["Opera", "Football"],
        row_payoffs=np.array([[2, 0], [0, 1]], dtype=float),
        col_payoffs=np.array([[1, 0], [0, 2]], dtype=float),
        known_ne=[
            (np.array([1., 0.]), np.array([1., 0.])),   # (Opera, Opera)
            (np.array([0., 1.]), np.array([0., 1.])),   # (Football, Football)
            (np.array([2/3, 1/3]), np.array([1/3, 2/3])),  # mixed NE
        ],
        description="Coordination with conflict of interest. Mixed NE payoffs are (2/3, 2/3).",
        game_type="coordination",
    ))

    # ------------------------------------------------------------------
    # 6. Stag Hunt
    #    (Stag, Stag) is Pareto-optimal; (Hare, Hare) is risk-dominant
    # ------------------------------------------------------------------
    games.append(NormalFormGame(
        name="Stag Hunt",
        row_actions=["Stag", "Hare"],
        col_actions=["Stag", "Hare"],
        row_payoffs=np.array([[4, 0], [2, 2]], dtype=float),
        col_payoffs=np.array([[4, 2], [0, 2]], dtype=float),
        known_ne=[
            (np.array([1., 0.]), np.array([1., 0.])),   # Pareto-optimal
            (np.array([0., 1.]), np.array([0., 1.])),   # risk-dominant
            (np.array([0.5, 0.5]), np.array([0.5, 0.5])),  # mixed NE
        ],
        description="Tests whether LLMs prefer Pareto-optimal vs risk-dominant equilibrium.",
        game_type="coordination",
    ))

    # ------------------------------------------------------------------
    # 7. Matching Pennies (zero-sum)
    #    Unique NE: (0.5 H, 0.5 T) for both players
    # ------------------------------------------------------------------
    games.append(NormalFormGame(
        name="Matching Pennies",
        row_actions=["Heads", "Tails"],
        col_actions=["Heads", "Tails"],
        row_payoffs=np.array([[1, -1], [-1, 1]], dtype=float),
        col_payoffs=np.array([[-1, 1], [1, -1]], dtype=float),
        known_ne=[(np.array([0.5, 0.5]), np.array([0.5, 0.5]))],
        description="Pure zero-sum. Only equilibrium is 50/50 mix. Tests randomization.",
        game_type="zero-sum",
    ))

    # ------------------------------------------------------------------
    # 8. Rock-Paper-Scissors (zero-sum)
    #    Unique NE: (1/3, 1/3, 1/3) for both
    # ------------------------------------------------------------------
    games.append(NormalFormGame(
        name="Rock-Paper-Scissors",
        row_actions=["Rock", "Paper", "Scissors"],
        col_actions=["Rock", "Paper", "Scissors"],
        row_payoffs=np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float),
        col_payoffs=np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=float),
        known_ne=[(np.array([1/3, 1/3, 1/3]), np.array([1/3, 1/3, 1/3]))],
        description="Zero-sum 3-action game. Tests uniform mixing and rock bias.",
        game_type="zero-sum",
    ))

    # ------------------------------------------------------------------
    # 9. Nash Bargaining — symmetric split-100
    #    Nash bargaining solution: (50, 50)
    # ------------------------------------------------------------------
    games.append(NormalFormGame(
        name="Nash Bargaining (Symmetric Split-100)",
        row_actions=[f"Demand {i*10}" for i in range(1, 10)],
        col_actions=[f"Demand {i*10}" for i in range(1, 10)],
        row_payoffs=np.array([
            [d1 if d1 + d2 <= 100 else 0
             for d2 in range(10, 100, 10)]
            for d1 in range(10, 100, 10)
        ], dtype=float),
        col_payoffs=np.array([
            [d2 if d1 + d2 <= 100 else 0
             for d2 in range(10, 100, 10)]
            for d1 in range(10, 100, 10)
        ], dtype=float),
        known_ne=[(np.array([0,0,0,0,1,0,0,0,0], dtype=float),
                   np.array([0,0,0,0,1,0,0,0,0], dtype=float))],  # demand 50 each
        description="Split-100 Nash demand game. NBS = (50, 50). Tests fairness/equality bias.",
        game_type="bargaining",
    ))

    # ------------------------------------------------------------------
    # 10. Nash Bargaining — asymmetric (outside options differ)
    #     Row has outside option 30, Col has outside option 10.
    #     NBS = (60, 40) when surplus=100
    # ------------------------------------------------------------------
    row_outside, col_outside = 30, 10
    surplus = 100
    games.append(NormalFormGame(
        name="Nash Bargaining (Asymmetric)",
        row_actions=[f"Demand {i*10}" for i in range(1, 10)],
        col_actions=[f"Demand {i*10}" for i in range(1, 10)],
        row_payoffs=np.array([
            [d1 if d1 + d2 <= surplus else row_outside
             for d2 in range(10, 100, 10)]
            for d1 in range(10, 100, 10)
        ], dtype=float),
        col_payoffs=np.array([
            [d2 if d1 + d2 <= surplus else col_outside
             for d2 in range(10, 100, 10)]
            for d1 in range(10, 100, 10)
        ], dtype=float),
        known_ne=[(np.array([0,0,0,0,0,1,0,0,0], dtype=float),
                   np.array([0,0,0,1,0,0,0,0,0], dtype=float))],  # demand 60 / 40
        description="Asymmetric outside options. NBS != 50/50. Tests whether LLM uses equality vs. rationality.",
        game_type="bargaining",
    ))

    # ------------------------------------------------------------------
    # 11. Chicken / Hawk-Dove
    #     Mixed NE: each plays Hawk w.p. 0.5
    # ------------------------------------------------------------------
    games.append(NormalFormGame(
        name="Chicken (Hawk-Dove)",
        row_actions=["Hawk (aggressive)", "Dove (yield)"],
        col_actions=["Hawk (aggressive)", "Dove (yield)"],
        row_payoffs=np.array([[-1, 2], [0, 1]], dtype=float),
        col_payoffs=np.array([[-1, 0], [2, 1]], dtype=float),
        known_ne=[
            (np.array([0., 1.]), np.array([1., 0.])),   # (Dove, Hawk)
            (np.array([1., 0.]), np.array([0., 1.])),   # (Hawk, Dove)
            (np.array([0.5, 0.5]), np.array([0.5, 0.5])),  # mixed NE
        ],
        description="Anti-coordination. Mixed NE = (0.5 Hawk, 0.5 Dove). Tests risk-aversion.",
        game_type="anti-coordination",
    ))

    # ------------------------------------------------------------------
    # 12. 3×3 General-Sum (no dominant strategies, unique mixed NE)
    # ------------------------------------------------------------------
    games.append(NormalFormGame(
        name="3x3 General-Sum Game",
        row_actions=["Action A", "Action B", "Action C"],
        col_actions=["Action X", "Action Y", "Action Z"],
        row_payoffs=np.array([[3, 0, 1], [1, 2, 0], [0, 1, 3]], dtype=float),
        col_payoffs=np.array([[2, 1, 0], [0, 3, 1], [1, 0, 2]], dtype=float),
        known_ne=[],  # computed numerically at runtime
        description="3×3 game with no pure-strategy NE. NE computed via LP solver.",
        game_type="general-sum",
    ))

    return games
