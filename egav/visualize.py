from typing import Dict, List, Tuple
import argparse
import json
from pathlib import Path
from collections import Counter

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


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE RESULTS VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────


def load_predictions(path: Path) -> List[Dict]:
    """Load prediction JSONL file."""
    preds = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))
    return preds


def compute_metrics(preds: List[Dict]) -> Dict:
    """Compute accuracy and abstention metrics."""
    correct = 0
    answered = 0
    abstained = 0
    clarify = 0
    total = len(preds)

    for p in preds:
        decision = p["decision"]
        if decision == "answer":
            answered += 1
            gold_answers = [a.lower().strip() for a in p.get("answers", [])]
            pred_text = p["final_text"].lower().strip()
            if any(pred_text in g or g in pred_text for g in gold_answers):
                correct += 1
        elif decision == "abstain":
            abstained += 1
        elif decision == "clarify":
            clarify += 1

    accuracy = correct / answered if answered > 0 else 0
    coverage = answered / total if total > 0 else 0

    return {
        "total": total,
        "answered": answered,
        "abstained": abstained,
        "clarify": clarify,
        "correct": correct,
        "accuracy": accuracy,
        "coverage": coverage,
        "f1_coverage": 2 * accuracy * coverage / (accuracy + coverage) if (accuracy + coverage) > 0 else 0,
    }


def plot_decision_pie(preds: List[Dict], ax):
    """Plot pie chart of decision distribution."""
    decisions = [p["decision"] for p in preds]
    counts = Counter(decisions)

    labels, sizes, color_list = [], [], []
    colors = {"answer": "#4CAF50", "abstain": "#FFC107", "clarify": "#2196F3"}

    for d in ["answer", "abstain", "clarify"]:
        if d in counts:
            labels.append(f"{d.capitalize()}\n({counts[d]})")
            sizes.append(counts[d])
            color_list.append(colors[d])

    ax.pie(sizes, labels=labels, colors=color_list, autopct="%1.1f%%", startangle=90)
    ax.set_title("Decision Distribution", fontsize=12, fontweight="bold")


def plot_verifier_histogram(preds: List[Dict], ax, tau_correct=0.5):
    """Plot histogram of verifier scores."""
    scores = [p["verifier_score"] for p in preds]
    ax.hist(scores, bins=30, color="#2196F3", alpha=0.7, edgecolor="black")
    ax.axvline(x=tau_correct, color="red", linestyle="--", linewidth=2, label=f"τ_correct={tau_correct}")
    ax.set_xlabel("Verifier Score")
    ax.set_ylabel("Count")
    ax.set_title("Verifier Score Distribution")
    ax.legend()


def plot_margin_histogram(preds: List[Dict], ax, tau_margin=0.1):
    """Plot histogram of margins."""
    margins = [p["margin"] for p in preds]
    ax.hist(margins, bins=30, color="#4CAF50", alpha=0.7, edgecolor="black")
    ax.axvline(x=tau_margin, color="red", linestyle="--", linewidth=2, label=f"τ_margin={tau_margin}")
    ax.set_xlabel("Margin (Top1 - Top2)")
    ax.set_ylabel("Count")
    ax.set_title("Margin Distribution")
    ax.legend()


def plot_decision_boxplot(preds: List[Dict], ax):
    """Box plot of verifier scores by decision type."""
    decisions = ["answer", "abstain", "clarify"]
    data, labels = [], []

    for d in decisions:
        scores = [p["verifier_score"] for p in preds if p["decision"] == d]
        if scores:
            data.append(scores)
            labels.append(d.capitalize())

    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = ["#4CAF50", "#FFC107", "#2196F3"]
        for patch, color in zip(bp["boxes"], colors[: len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax.set_ylabel("Verifier Score")
    ax.set_title("Verifier Score by Decision")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)


def plot_correct_scatter(preds: List[Dict], ax):
    """Scatter plot: correct vs incorrect answers."""
    correct_v, correct_m = [], []
    incorrect_v, incorrect_m = [], []

    for p in preds:
        if p["decision"] == "answer":
            gold_answers = [a.lower().strip() for a in p.get("answers", [])]
            pred_text = p["final_text"].lower().strip()
            is_correct = any(pred_text in g or g in pred_text for g in gold_answers)

            if is_correct:
                correct_v.append(p["verifier_score"])
                correct_m.append(p["margin"])
            else:
                incorrect_v.append(p["verifier_score"])
                incorrect_m.append(p["margin"])

    ax.scatter(correct_v, correct_m, c="#4CAF50", alpha=0.6, label=f"Correct ({len(correct_v)})", s=30)
    ax.scatter(incorrect_v, incorrect_m, c="#F44336", alpha=0.6, label=f"Incorrect ({len(incorrect_v)})", s=30)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Verifier Score")
    ax.set_ylabel("Margin")
    ax.set_title("Correct vs Incorrect")
    ax.legend()


def plot_combined_score_hist(preds: List[Dict], ax):
    """Histogram of combined scores."""
    scores = [p["final_score"] for p in preds]
    ax.hist(scores, bins=30, color="#FF9800", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Combined Score")
    ax.set_ylabel("Count")
    ax.set_title("Combined Score Distribution")


def print_examples(preds: List[Dict], n: int = 3):
    """Print example predictions for each decision type."""
    print("\n" + "=" * 80)
    print("EXAMPLE PREDICTIONS")
    print("=" * 80)

    for decision in ["answer", "abstain", "clarify"]:
        examples = [p for p in preds if p["decision"] == decision][:n]
        if examples:
            print(f"\n{'─' * 40}")
            print(f"📌 {decision.upper()} examples:")
            print(f"{'─' * 40}")
            for i, ex in enumerate(examples, 1):
                q = ex.get("question", "")[:80]
                print(f"\n  [{i}] Q: {q}...")
                print(f"      Gold: {ex.get('answers', ['N/A'])[:2]}")
                if decision == "answer":
                    print(f"      Pred: {ex['final_text']}")
                print(f"      V={ex['verifier_score']:.3f}, M={ex['margin']:.3f}")


def visualize_inference(pred_path: Path, output_dir: Path = None, show: bool = True):
    """Main visualization for inference results."""
    apply_style()
    preds = load_predictions(pred_path)
    metrics = compute_metrics(preds)

    # Print metrics table
    print("\n" + "=" * 50)
    print("📊 EGAV INFERENCE RESULTS")
    print("=" * 50)
    print(f"  Total examples:     {metrics['total']}")
    print(f"  Answered:           {metrics['answered']} ({100*metrics['coverage']:.1f}%)")
    print(f"  Abstained:          {metrics['abstained']}")
    print(f"  Clarify:            {metrics['clarify']}")
    print(f"  Correct (answered): {metrics['correct']}")
    print(f"  ─────────────────────────────────────")
    print(f"  Accuracy:           {100*metrics['accuracy']:.2f}%")
    print(f"  Coverage:           {100*metrics['coverage']:.2f}%")
    print(f"  F1 (Acc×Cov):       {100*metrics['f1_coverage']:.2f}%")
    print("=" * 50)

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("EGAV Pipeline Results", fontsize=16, fontweight="bold", y=0.98)

    ax1 = fig.add_subplot(2, 3, 1)
    plot_decision_pie(preds, ax1)

    ax2 = fig.add_subplot(2, 3, 2)
    plot_verifier_histogram(preds, ax2)

    ax3 = fig.add_subplot(2, 3, 3)
    plot_margin_histogram(preds, ax3)

    ax4 = fig.add_subplot(2, 3, 4)
    plot_decision_boxplot(preds, ax4)

    ax5 = fig.add_subplot(2, 3, 5)
    plot_correct_scatter(preds, ax5)

    ax6 = fig.add_subplot(2, 3, 6)
    plot_combined_score_hist(preds, ax6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save or show
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / "egav_results.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"\n📈 Saved: {fig_path}")

        metrics_path = output_dir / "metrics.json"
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)
        print(f"📋 Saved: {metrics_path}")

    if show:
        plt.show()

    print_examples(preds)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Visualize EGAV inference results")
    parser.add_argument("--predictions", "-p", type=str, required=True, help="Path to predictions JSONL")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--no-show", action="store_true", help="Don't display plot (save only)")
    args = parser.parse_args()

    visualize_inference(
        Path(args.predictions),
        Path(args.output_dir) if args.output_dir else None,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
