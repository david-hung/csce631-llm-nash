"""
Nash equilibrium computation for two-player normal-form games.
Uses nashpy (support enumeration + vertex enumeration) and scipy LP as fallback.
"""
import numpy as np
import nashpy as nash
from scipy.optimize import linprog
from typing import List, Tuple

from src.games.normal_form import NormalFormGame


def compute_nash_equilibria(game: NormalFormGame) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of Nash equilibria via nashpy support enumeration."""
    rps = nash.Game(game.row_payoffs, game.col_payoffs)
    equilibria = []
    try:
        for eq in rps.support_enumeration():
            r, c = eq
            if _is_valid_ne(game, r, c):
                equilibria.append((r, c))
    except Exception:
        pass
    return equilibria if equilibria else support_enumeration_nash(game)


def support_enumeration_nash(game: NormalFormGame) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Manual support enumeration fallback."""
    from itertools import combinations
    m, n = game.num_row_actions, game.num_col_actions
    equilibria = []

    for r_support_size in range(1, m + 1):
        for c_support_size in range(1, n + 1):
            for r_sup in combinations(range(m), r_support_size):
                for c_sup in combinations(range(n), c_support_size):
                    r_strat = _find_indifferent_mix(game.row_payoffs, list(r_sup), list(c_sup))
                    c_strat = _find_indifferent_mix(game.col_payoffs.T, list(c_sup), list(r_sup))
                    if r_strat is not None and c_strat is not None:
                        if _is_valid_ne(game, r_strat, c_strat, tol=1e-6):
                            equilibria.append((r_strat, c_strat))

    return equilibria


def _find_indifferent_mix(payoff_matrix: np.ndarray, own_support: list, opp_support: list) -> np.ndarray | None:
    """
    Find a mixed strategy for the 'row' player over own_support such that all
    actions in opp_support yield equal expected payoff (indifference condition).
    """
    n_opp = len(opp_support)
    n_own = len(own_support)

    if n_opp == 1:
        vec = np.zeros(payoff_matrix.shape[1])
        vec[opp_support[0]] = 1.0
        return vec

    # Build linear system for indifference
    sub = payoff_matrix[np.ix_(opp_support, own_support)]
    # sub[i] - sub[0] = 0 for i>0, plus sum = 1
    A = np.vstack([sub[1:] - sub[0], np.ones((1, n_own))])
    b = np.zeros(n_opp)
    b[-1] = 1.0

    try:
        x, res, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        if rank < A.shape[1]:
            return None
        if np.any(x < -1e-8):
            return None
        x = np.clip(x, 0, None)
        x /= x.sum()
        full = np.zeros(payoff_matrix.shape[1])
        for i, idx in enumerate(own_support):
            full[idx] = x[i]
        return full
    except Exception:
        return None


def _is_valid_ne(game: NormalFormGame, row_strat: np.ndarray, col_strat: np.ndarray, tol: float = 1e-4) -> bool:
    """Verify that (row_strat, col_strat) is an epsilon-Nash equilibrium."""
    if np.any(row_strat < -tol) or np.any(col_strat < -tol):
        return False
    if abs(row_strat.sum() - 1) > tol or abs(col_strat.sum() - 1) > tol:
        return False
    # Row player best response value
    row_vals = game.row_payoffs @ col_strat
    br_row_val = row_vals.max()
    current_row_val = row_strat @ row_vals
    if br_row_val - current_row_val > tol:
        return False
    # Col player best response value
    col_vals = row_strat @ game.col_payoffs
    br_col_val = col_vals.max()
    current_col_val = col_vals @ col_strat
    if br_col_val - current_col_val > tol:
        return False
    return True
