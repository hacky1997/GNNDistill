import json
import math
import random
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_device(prefer_mps: bool = True):
    try:
        import torch

        if prefer_mps and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except Exception:
        return "cpu"


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = " ".join(text.split())
    return text


def compute_em_f1(prediction: str, ground_truths: List[str]) -> Tuple[float, float]:
    if not ground_truths:
        return 0.0, 0.0
    em = max(exact_match_score(prediction, gt) for gt in ground_truths)
    f1 = max(f1_score(prediction, gt) for gt in ground_truths)
    return em, f1


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def char_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = list(normalize_text(prediction))
    gold_tokens = list(normalize_text(ground_truth))
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def softmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    max_score = max(scores)
    exps = [math.exp(s - max_score) for s in scores]
    denom = sum(exps)
    if denom == 0:
        return [0.0 for _ in scores]
    return [e / denom for e in exps]


def entropy(probs: List[float]) -> float:
    total = 0.0
    for p in probs:
        if p > 0:
            total -= p * math.log(p + 1e-12)
    return total


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for key, value in d.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            out.update(flatten_dict(value, full_key))
        else:
            out[full_key] = value
    return out


def config_to_dict(config) -> Dict[str, Any]:
    return flatten_dict(asdict(config))
