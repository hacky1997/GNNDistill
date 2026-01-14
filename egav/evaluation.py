import math
import random
from typing import Dict, List, Tuple

from egav.utils import compute_em_f1


def compute_metrics_from_predictions(preds: Dict[str, str], dataset) -> Dict[str, float]:
    total_em = 0.0
    total_f1 = 0.0
    count = 0
    for ex in dataset:
        qid = ex["id"]
        answers = ex.get("answers", {})
        golds = answers.get("text", []) if answers else []
        pred = preds.get(qid, "")
        em, f1 = compute_em_f1(pred, golds)
        total_em += em
        total_f1 += f1
        count += 1
    if count == 0:
        return {"em": 0.0, "f1": 0.0}
    return {"em": total_em / count, "f1": total_f1 / count}


def evaluate_abstention(pred_rows: List[Dict]) -> Dict[str, float]:
    total = 0
    answered = 0
    em_total = 0.0
    f1_total = 0.0
    em_answered = 0.0
    f1_answered = 0.0

    for row in pred_rows:
        total += 1
        pred = row.get("final_text", "") if row.get("decision") == "answer" else ""
        golds = row.get("answers", [])
        em, f1 = compute_em_f1(pred, golds)
        em_total += em
        f1_total += f1
        if row.get("decision") == "answer":
            answered += 1
            em_answered += em
            f1_answered += f1

    coverage = answered / total if total else 0.0
    metrics = {
        "coverage": coverage,
        "em_overall": em_total / total if total else 0.0,
        "f1_overall": f1_total / total if total else 0.0,
        "em_answered": em_answered / answered if answered else 0.0,
        "f1_answered": f1_answered / answered if answered else 0.0,
    }
    return metrics


def risk_coverage_curve(rows: List[Dict], score_key: str = "verifier_score") -> Tuple[List[float], List[float]]:
    sorted_rows = sorted(rows, key=lambda r: r.get(score_key, 0.0), reverse=True)
    risks = []
    coverages = []
    total = len(sorted_rows)
    cum_risk = 0.0
    for i, row in enumerate(sorted_rows, start=1):
        pred = row.get("final_text", "")
        golds = row.get("answers", [])
        _, f1 = compute_em_f1(pred, golds)
        risk = 1.0 - f1
        cum_risk += risk
        coverage = i / total
        risks.append(cum_risk / i)
        coverages.append(coverage)
    return coverages, risks


def aurc(coverages: List[float], risks: List[float]) -> float:
    if not coverages:
        return 0.0
    area = 0.0
    prev_c = 0.0
    prev_r = risks[0]
    for c, r in zip(coverages, risks):
        area += (c - prev_c) * (r + prev_r) / 2.0
        prev_c = c
        prev_r = r
    return area


def bootstrap_ci(metrics_func, rows: List[Dict], samples: int, seed: int = 7):
    random.seed(seed)
    values = []
    n = len(rows)
    for _ in range(samples):
        sample = [rows[random.randint(0, n - 1)] for _ in range(n)]
        values.append(metrics_func(sample))
    return values
