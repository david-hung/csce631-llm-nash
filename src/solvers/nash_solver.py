"""
Nash equilibrium computation for two-player normal-form games.

Primary method: nashpy support enumeration (handles all standard cases).
Fallback: manual support enumeration (slower, used when nashpy yields nothing).
"""
import numpy as np
import nashpy as nash
from itertools import combinations
from typing import List, Tuple

from src.games.normal_form import NormalFormGame


def compute_nash_equilibria(game: NormalFormGame) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of Nash equilibria via nashpy, with manual fallback."""
    rps = nash.Game(game.row_payoffs, game.col_payoffs)
    equilibria = []
    try:
        for eq in rps.support_enumeration():
            r, c = np.array(eq[0]), np.array(eq[1])
            if _is_valid_ne(game, r, c):
                equilibria.append((r, c))
    except Exception:
        pass

    if not equilibria:
        equilibria = _manual_support_enumeration(game)

    return equilibria


def _manual_support_enumeration(game: NormalFormGame) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Enumerate all support pairs and solve for indifference conditions.

    For a given (row_support, col_support):
      - col player's mix q over col_support must make row player indifferent over row_support:
          row_payoffs[row_support, :][:, col_support] @ q = constant
      - row player's mix p over row_support must make col player indifferent over col_support:
          col_payoffs.T[col_support, :][:, row_support] @ p = constant
    """
    m, n = game.num_row_actions, game.num_col_actions
    equilibria = []

    for r_size in range(1, m + 1):
        for c_size in range(1, n + 1):
            for r_sup in combinations(range(m), r_size):
                for c_sup in combinations(range(n), c_size):
                    r_sup, c_sup = list(r_sup), list(c_sup)

                    # Solve for col player's mix (makes row player indifferent over r_sup)
                    A_row = game.row_payoffs[np.ix_(r_sup, c_sup)]  # shape (|r_sup|, |c_sup|)
                    q = _solve_indifference(A_row, n, c_sup)

                    # Solve for row player's mix (makes col player indifferent over c_sup)
                    A_col = game.col_payoffs.T[np.ix_(c_sup, r_sup)]  # shape (|c_sup|, |r_sup|)
                    p = _solve_indifference(A_col, m, r_sup)

                    if p is not None and q is not None:
                        if _is_valid_ne(game, p, q, tol=1e-6):
                            equilibria.append((p, q))

    return equilibria


def _solve_indifference(submatrix: np.ndarray, n_total: int, support: list) -> np.ndarray | None:
    """
    Find a probability vector over `support` (length n_total, zeros elsewhere) such that
    all rows of `submatrix` yield equal expected value against that vector.

    submatrix: shape (n_opp, n_own) — payoffs for opponent's actions (rows)
               against our support actions (cols)
    """
    n_opp, n_own = submatrix.shape

    if n_own == 1:
        # Single action in support — trivially plays it with prob 1
        vec = np.zeros(n_total)
        vec[support[0]] = 1.0
        return vec

    # Build linear system: indifference across all opponent actions + sum-to-1
    # (submatrix[i] - submatrix[0]) @ x = 0  for i = 1..n_opp-1
    # sum(x) = 1
    A = np.vstack([submatrix[1:] - submatrix[0], np.ones((1, n_own))])
    b = np.zeros(n_opp)
    b[-1] = 1.0

    try:
        x, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if rank < n_own:
        return None
    if np.any(x < -1e-8):
        return None

    x = np.clip(x, 0, None)
    if x.sum() < 1e-10:
        return None
    x /= x.sum()

    full = np.zeros(n_total)
    for i, idx in enumerate(support):
        full[idx] = x[i]
    return full


def _is_valid_ne(game: NormalFormGame, row_strat: np.ndarray,
                 col_strat: np.ndarray, tol: float = 1e-4) -> bool:
    """Verify (row_strat, col_strat) is an epsilon-Nash equilibrium."""
    if np.any(row_strat < -tol) or np.any(col_strat < -tol):
        return False
    if abs(row_strat.sum() - 1) > tol or abs(col_strat.sum() - 1) > tol:
        return False

    row_vals = game.row_payoffs @ col_strat
    if row_vals.max() - row_strat @ row_vals > tol:
        return False

    col_vals = row_strat @ game.col_payoffs
    if col_vals.max() - col_vals @ col_strat > tol:
        return False

    return True
