import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from egav.data_mlqa import load_mlqa
from egav.utils import ensure_dir, softmax, get_device


def generate_nbest(
    model,
    tokenizer,
    question: str,
    context: str,
    max_length: int,
    doc_stride: int,
    n_best_size: int,
    max_answer_length: int,
    device,
):
    tokenized = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )
    offset_mapping = tokenized.pop("offset_mapping")
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    candidates = []
    model.eval()
    with torch.no_grad():
        for i in range(len(sample_mapping)):
            inputs = {k: v[i : i + 1].to(device) for k, v in tokenized.items()}
            outputs = model(**inputs)
            start_logits = outputs.start_logits.squeeze(0).detach().cpu().numpy()
            end_logits = outputs.end_logits.squeeze(0).detach().cpu().numpy()

            offsets = offset_mapping[i]
            sequence_ids = tokenized.sequence_ids(i)
            start_indexes = np.argsort(start_logits)[-n_best_size:][::-1]
            end_indexes = np.argsort(end_logits)[-n_best_size:][::-1]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offsets) or end_index >= len(offsets):
                        continue
                    if sequence_ids[start_index] != 1 or sequence_ids[end_index] != 1:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    start_char, _ = offsets[start_index]
                    _, end_char = offsets[end_index]
                    text = context[start_char:end_char]
                    candidates.append(
                        {
                            "span_text": text,
                            "start_char": int(start_char),
                            "end_char": int(end_char),
                            "start_logit": float(start_logits[start_index]),
                            "end_logit": float(end_logits[end_index]),
                            "span_logit_sum": float(start_logits[start_index] + end_logits[end_index]),
                        }
                    )

    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda x: x["span_logit_sum"], reverse=True)
    candidates = candidates[:n_best_size]
    probs = softmax([c["span_logit_sum"] for c in candidates])
    for idx, c in enumerate(candidates):
        c["span_prob"] = probs[idx]
        c["rank"] = idx
    return candidates


def dump_candidates(
    dataset,
    model,
    tokenizer,
    output_path: Path,
    max_length: int,
    doc_stride: int,
    n_best_size: int,
    max_answer_length: int,
    device,
    limit: Optional[int] = None,
):
    ensure_dir(output_path.parent)
    rows = []
    for i, ex in enumerate(dataset):
        if limit and i >= limit:
            break
        candidates = generate_nbest(
            model,
            tokenizer,
            ex["question"],
            ex["context"],
            max_length,
            doc_stride,
            n_best_size,
            max_answer_length,
            device,
        )
        rows.append(
            {
                "id": ex["id"],
                "question": ex["question"],
                "context": ex["context"],
                "answers": ex.get("answers", {}),
                "lang": ex.get("lang", None),
                "candidates": candidates,
            }
        )
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def oracle_at_k(candidates_rows: List[Dict], k: int, threshold: float = 0.5) -> float:
    from egav.utils import f1_score

    hits = 0
    total = 0
    for row in candidates_rows:
        answers = row.get("answers", {})
        golds = answers.get("text", []) if answers else []
        if not golds:
            continue
        total += 1
        for cand in row["candidates"][:k]:
            if max(f1_score(cand["span_text"], g) for g in golds) >= threshold:
                hits += 1
                break
    return hits / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--languages", type=str, default="en")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--n_best_size", type=int, default=20)
    parser.add_argument("--max_answer_length", type=int, default=30)
    args = parser.parse_args()

    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    ds = load_mlqa(languages=args.languages.split(","))
    dataset = ds[args.split]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    device = get_device()
    model.to(device)

    dump_candidates(
        dataset,
        model,
        tokenizer,
        Path(args.output),
        args.max_length,
        args.doc_stride,
        args.n_best_size,
        args.max_answer_length,
        device,
        limit=args.limit if args.limit > 0 else None,
    )


if __name__ == "__main__":
    main()
