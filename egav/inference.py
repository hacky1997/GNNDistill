import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from egav.config import default_config
from egav.graph_features import compute_features_for_candidates
from egav.ner import extract_mentions, mentions_in_span
from egav.train_verifier import FeatureScaler
from egav.gnn_verifier import MLPVerifier
from egav.utils import read_jsonl, ensure_dir


def load_verifier(checkpoint_path: Path, in_dim: int, hidden_dim: int, dropout: float):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    scaler = FeatureScaler()
    scaler.load_state_dict(ckpt["scaler"])
    in_dim = len(ckpt.get("feature_names", [])) or in_dim
    model = MLPVerifier(in_dim, hidden_dim, dropout)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, scaler


def candidate_entity_signatures(candidates: List[Dict], c_mentions):
    sigs = []
    for cand in candidates:
        mentions = mentions_in_span(c_mentions, cand["start_char"], cand["end_char"])
        sigs.append({m.norm for m in mentions if m.norm})
    return sigs


def run_inference(
    candidates_path: Path,
    verifier_ckpt: Path,
    output_path: Path,
    gamma: float,
    tau_correct: float,
    tau_margin: float,
    lang: str,
    cfg,
):
    ensure_dir(output_path.parent)
    model, scaler = load_verifier(verifier_ckpt, cfg.verifier.feature_dim, cfg.verifier.hidden_dim, cfg.verifier.dropout)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)

    outputs = []
    for row in read_jsonl(candidates_path):
        q_mentions = extract_mentions(row["question"], lang=lang)
        c_mentions = extract_mentions(row["context"], lang=lang)
        feats = compute_features_for_candidates(
            row["question"],
            row["context"],
            row["candidates"],
            q_mentions,
            c_mentions,
        )
        feats = scaler.transform(np.array(feats, dtype=np.float32))
        with torch.no_grad():
            scores = model(torch.tensor(feats).to(device)).cpu().numpy()
        scores = np.clip(scores, 0.0, 1.0)

        for cand, s in zip(row["candidates"], scores):
            cand["verifier_score"] = float(s)
            cand["combined_score"] = float(cand.get("span_logit_sum", 0.0) + gamma * s)

        sorted_cands = sorted(row["candidates"], key=lambda x: x["combined_score"], reverse=True)
        top1 = sorted_cands[0]
        top2 = sorted_cands[1] if len(sorted_cands) > 1 else None
        margin = top1["combined_score"] - top2["combined_score"] if top2 else top1["combined_score"]

        decision = "abstain"
        if top1["verifier_score"] >= tau_correct and margin >= tau_margin:
            decision = "answer"
        else:
            sigs = candidate_entity_signatures(sorted_cands[:2], c_mentions)
            if top2 and sigs and len(sigs) == 2:
                if sigs[0] and sigs[1] and sigs[0].isdisjoint(sigs[1]):
                    decision = "clarify"

        outputs.append(
            {
                "id": row["id"],
                "decision": decision,
                "final_text": top1["span_text"] if decision == "answer" else "",
                "final_score": top1["combined_score"],
                "verifier_score": top1["verifier_score"],
                "margin": margin,
                "top1": top1,
                "top2": top2,
                "answers": row.get("answers", {}).get("text", []),
                "question": row.get("question", ""),
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        for out in outputs:
            f.write(json.dumps(out, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, required=True)
    parser.add_argument("--verifier", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--tau_correct", type=float, default=0.5)
    parser.add_argument("--tau_margin", type=float, default=0.1)
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()

    cfg = default_config()
    run_inference(
        Path(args.candidates),
        Path(args.verifier),
        Path(args.output),
        args.gamma,
        args.tau_correct,
        args.tau_margin,
        args.lang,
        cfg,
    )


if __name__ == "__main__":
    main()
