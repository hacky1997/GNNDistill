import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from egav.config import default_config
from egav.graph_features import compute_features_for_candidates, feature_names
from egav.ner import extract_mentions
from egav.gnn_verifier import MLPVerifier
from egav.utils import ensure_dir, f1_score, read_jsonl, set_seed, get_device


class FeatureScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-8

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    def load_state_dict(self, state: Dict):
        self.mean = np.array(state["mean"])
        self.std = np.array(state["std"])


class VerifierDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, qids: List[str]):
        self.features = features
        self.labels = labels
        self.qids = qids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.qids[idx]


def build_verifier_rows(candidates_path: Path, lang: str = "en") -> List[Dict]:
    rows = []
    for row in read_jsonl(candidates_path):
        q_mentions = extract_mentions(row["question"], lang=lang)
        c_mentions = extract_mentions(row["context"], lang=lang)
        features = compute_features_for_candidates(
            row["question"],
            row["context"],
            row["candidates"],
            q_mentions,
            c_mentions,
        )
        gold_texts = row.get("answers", {}).get("text", [])
        for cand, feats in zip(row["candidates"], features):
            y = 0.0
            if gold_texts:
                y = max(f1_score(cand["span_text"], gt) for gt in gold_texts)
            rows.append(
                {
                    "id": row["id"],
                    "span_text": cand["span_text"],
                    "rank": cand.get("rank", 0),
                    "features": feats,
                    "y": float(y),
                    "y_hard": 1 if y >= 0.5 else 0,
                }
            )
    return rows


def pairwise_ranking_loss(scores, labels, qids, margin: float, max_pairs: int = 20):
    loss = 0.0
    count = 0
    unique_qids = list(set(qids))
    for qid in unique_qids:
        indices = [i for i, q in enumerate(qids) if q == qid]
        if len(indices) < 2:
            continue
        pos = [i for i in indices if labels[i] >= 0.5]
        neg = [i for i in indices if labels[i] < 0.5]
        if not pos or not neg:
            continue
        random.shuffle(pos)
        random.shuffle(neg)
        pairs = 0
        for p in pos:
            for n in neg:
                loss = loss + torch.relu(margin - (scores[p] - scores[n]))
                count += 1
                pairs += 1
                if pairs >= max_pairs:
                    break
            if pairs >= max_pairs:
                break
    if count == 0:
        return torch.tensor(0.0, device=scores.device)
    return loss / count


def train_verifier(cfg, candidates_path: Path, output_dir: Path, lang: str = "en"):
    set_seed(cfg.training.seed)
    rows = build_verifier_rows(candidates_path, lang=lang)
    feats = np.array([r["features"] for r in rows], dtype=np.float32)
    labels = np.array([r["y"] for r in rows], dtype=np.float32)
    qids = [r["id"] for r in rows]

    scaler = FeatureScaler()
    scaler.fit(feats)
    feats = scaler.transform(feats)

    dataset = VerifierDataset(feats, labels, qids)
    loader = DataLoader(dataset, batch_size=cfg.verifier.batch_size, shuffle=True)

    device = get_device()
    feature_dim = feats.shape[1]
    model = MLPVerifier(feature_dim, cfg.verifier.hidden_dim, cfg.verifier.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.verifier.learning_rate, weight_decay=cfg.verifier.weight_decay)
    huber = nn.SmoothL1Loss(beta=cfg.verifier.huber_delta)

    model.train()
    for epoch in range(cfg.verifier.num_train_epochs):
        total_loss = 0.0
        for batch in loader:
            x, y, qid = batch
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            reg_loss = huber(scores, y)
            rank_loss = pairwise_ranking_loss(scores, y, qid, cfg.verifier.rank_margin)
            loss = reg_loss + cfg.verifier.lambda_rank * rank_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(loader), 1)
        print(f"epoch {epoch + 1} loss {avg_loss:.4f}")

    ensure_dir(output_dir)
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler": scaler.state_dict(),
            "feature_names": feature_names(),
        },
        output_dir / "verifier_mlp.pt",
    )
    with (output_dir / "verifier_rows.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()

    cfg = default_config()
    train_verifier(cfg, Path(args.candidates), Path(args.output), args.lang)


if __name__ == "__main__":
    main()
