"""
Visualization utilities for LLM Nash analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import List, Dict


def plot_exploitability_heatmap(results: Dict, save_path: str | None = None):
    """
    Heatmap of exploitability per game (rows) × temperature (cols).

    results: {game_name: {temperature: exploitability_value}}
    """
    games = list(results.keys())
    temps = sorted({t for v in results.values() for t in v.keys()})

    data = np.array([[results[g].get(t, np.nan) for t in temps] for g in games])

    fig, ax = plt.subplots(figsize=(6, max(4, len(games) * 0.55)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=0)
    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([f"T={t}" for t in temps])
    ax.set_yticks(range(len(games)))
    ax.set_yticklabels(games, fontsize=8)
    plt.colorbar(im, ax=ax, label="Exploitability")
    ax.set_title("LLM Exploitability vs. Nash Equilibrium")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig, ax


def plot_strategy_comparison(game_name: str, actions: List[str],
                             ne_strategy: np.ndarray,
                             llm_strategies: Dict[float, np.ndarray],
                             player: int = 1,
                             save_path: str | None = None):
    """
    Bar chart comparing Nash equilibrium strategy vs. LLM strategies at each temperature.
    """
    temps = sorted(llm_strategies.keys())
    n_actions = len(actions)
    x = np.arange(n_actions)
    width = 0.15

    fig, ax = plt.subplots(figsize=(max(6, n_actions * 1.5), 4))

    ax.bar(x - (len(temps) / 2) * width, ne_strategy, width, label="Nash NE", color="steelblue", alpha=0.9)
    colors = ["tomato", "orange", "purple"]
    for i, t in enumerate(temps):
        ax.bar(x + (i - len(temps) / 2 + 1) * width, llm_strategies[t], width,
               label=f"LLM T={t}", color=colors[i % len(colors)], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(actions, rotation=20, ha="right")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{game_name} — Player {player} Strategy")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig, ax


def plot_bias_summary(bias_scores: Dict[str, float], save_path: str | None = None):
    """
    Horizontal bar chart of per-game cooperation/deviation bias.
    bias_scores: {game_name: signed_deviation_from_ne}
    Positive = more cooperative/fair than NE, negative = more competitive.
    """
    games = list(bias_scores.keys())
    scores = [bias_scores[g] for g in games]
    colors = ["seagreen" if s > 0 else "tomato" for s in scores]

    fig, ax = plt.subplots(figsize=(7, max(4, len(games) * 0.5)))
    ax.barh(games, scores, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Signed Deviation from Nash Strategy (cooperation bias > 0)")
    ax.set_title("Systematic Biases in LLM Strategic Behavior")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig, ax


def plot_exploitability_by_temperature(results: Dict, game_name: str, save_path: str | None = None):
    """Line plot of exploitability vs temperature for a single game."""
    temps = sorted(results[game_name].keys())
    vals = [results[game_name][t] for t in temps]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(temps, vals, marker="o", color="tomato")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Exploitability")
    ax.set_title(f"{game_name}")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig, ax
