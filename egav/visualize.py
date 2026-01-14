from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from egav.config import default_config


def apply_style(cfg=None):
    if cfg is None:
        cfg = default_config()
    plt.rcParams.update(
        {
            "font.family": cfg.plot.font_family,
            "font.size": cfg.plot.base_font_size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": cfg.plot.dpi,
        }
    )


def plot_risk_coverage(curves: Dict[str, Tuple[List[float], List[float]]], output_path: str):
    apply_style()
    fig, ax = plt.subplots()
    for label, (coverage, risk) in curves.items():
        ax.plot(np.array(coverage) * 100, [1 - r for r in risk], label=label, linewidth=2)
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("F1 (overall)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)


def plot_coverage_answered_f1(curves: Dict[str, Tuple[List[float], List[float]]], output_path: str):
    apply_style()
    fig, ax = plt.subplots()
    for label, (coverage, f1) in curves.items():
        ax.plot(np.array(coverage) * 100, f1, label=label, linewidth=2)
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("F1 (answered-only)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)


def plot_reliability_diagram(bins: List[float], accuracies: List[float], output_path: str):
    apply_style()
    fig, ax = plt.subplots()
    ax.plot(bins, accuracies, marker="o", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Predicted correctness")
    ax.set_ylabel("Empirical accuracy")
    fig.tight_layout()
    fig.savefig(output_path)


def plot_oracle_k(oracle_scores: Dict[int, float], output_path: str):
    apply_style()
    ks = sorted(oracle_scores.keys())
    vals = [oracle_scores[k] for k in ks]
    fig, ax = plt.subplots()
    ax.bar([str(k) for k in ks], vals, color="#333333")
    ax.set_xlabel("K")
    ax.set_ylabel("Oracle@K")
    fig.tight_layout()
    fig.savefig(output_path)


def plot_error_buckets(bucket_scores: Dict[str, float], output_path: str):
    apply_style()
    labels = list(bucket_scores.keys())
    vals = list(bucket_scores.values())
    fig, ax = plt.subplots()
    ax.barh(labels, vals, color="#555555")
    ax.set_xlabel("Count (%)")
    fig.tight_layout()
    fig.savefig(output_path)


def plot_ablation(methods: List[str], scores: List[float], output_path: str):
    apply_style()
    fig, ax = plt.subplots()
    ax.bar(range(len(methods)), scores, color="#444444")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("F1")
    fig.tight_layout()
    fig.savefig(output_path)


def plot_language_heatmap(langs: List[str], metrics: np.ndarray, output_path: str):
    apply_style()
    fig, ax = plt.subplots()
    im = ax.imshow(metrics, cmap="Greys")
    ax.set_xticks(range(metrics.shape[1]))
    ax.set_yticks(range(metrics.shape[0]))
    ax.set_yticklabels(langs)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)


def plot_score_distributions(correct_scores: List[float], incorrect_scores: List[float], output_path: str):
    apply_style()
    fig, ax = plt.subplots()
    ax.hist(correct_scores, bins=30, alpha=0.7, label="Correct", color="#222222")
    ax.hist(incorrect_scores, bins=30, alpha=0.5, label="Incorrect", color="#777777")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)


def plot_evidence_graph(graph: nx.DiGraph, node_labels: Dict[int, str], highlight_path: List[int], output_path: str):
    apply_style()
    pos = nx.spring_layout(graph, seed=7)
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=120, node_color="#999999")
    nx.draw_networkx_edges(graph, pos, ax=ax, width=0.8, alpha=0.6)
    if highlight_path:
        path_edges = list(zip(highlight_path[:-1], highlight_path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, ax=ax, width=2.0, edge_color="#111111")
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path)


def plot_ambiguity_panel(candidate_a: Dict, candidate_b: Dict, output_path: str):
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    text = (
        "Candidate 1\n"
        f"{candidate_a.get('span_text', '')}\n"
        f"entities: {candidate_a.get('entities', '')}\n\n"
        "Candidate 2\n"
        f"{candidate_b.get('span_text', '')}\n"
        f"entities: {candidate_b.get('entities', '')}"
    )
    ax.text(0.02, 0.98, text, va="top", ha="left")
    fig.tight_layout()
    fig.savefig(output_path)
