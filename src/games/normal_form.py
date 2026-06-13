from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional


@dataclass
class NormalFormGame:
    """Two-player normal-form game with named actions and known Nash equilibria."""
    name: str
    row_actions: List[str]
    col_actions: List[str]
    row_payoffs: np.ndarray  # shape (|row_actions|, |col_actions|)
    col_payoffs: np.ndarray  # shape (|row_actions|, |col_actions|)
    # Known Nash equilibria: list of (row_strategy, col_strategy) probability vectors
    known_ne: List[tuple] = field(default_factory=list)
    description: str = ""
    game_type: str = "general-sum"  # "zero-sum", "coordination", "social-dilemma", etc.
    prompt_template: Optional[str] = None  # custom LLM prompt; None uses default

    @property
    def num_row_actions(self) -> int:
        return len(self.row_actions)

    @property
    def num_col_actions(self) -> int:
        return len(self.col_actions)

    def payoff(self, row_idx: int, col_idx: int) -> tuple:
        return float(self.row_payoffs[row_idx, col_idx]), float(self.col_payoffs[row_idx, col_idx])

    def expected_payoff(self, row_strategy: np.ndarray, col_strategy: np.ndarray) -> tuple:
        r = float(row_strategy @ self.row_payoffs @ col_strategy)
        c = float(row_strategy @ self.col_payoffs @ col_strategy)
        return r, c

    def best_response_row(self, col_strategy: np.ndarray) -> np.ndarray:
        """Pure best response for row player against a given col mixed strategy."""
        values = self.row_payoffs @ col_strategy
        br = np.zeros(self.num_row_actions)
        br[np.argmax(values)] = 1.0
        return br

    def best_response_col(self, row_strategy: np.ndarray) -> np.ndarray:
        values = row_strategy @ self.col_payoffs
        br = np.zeros(self.num_col_actions)
        br[np.argmax(values)] = 1.0
        return br

    def __repr__(self) -> str:
        return f"NormalFormGame(name={self.name!r}, actions={self.row_actions}×{self.col_actions})"
