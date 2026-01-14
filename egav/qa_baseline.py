import argparse
import inspect
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from egav.config import default_config
from egav.data_mlqa import load_mlqa, prepare_train_features, prepare_validation_features
from egav.evaluation import compute_metrics_from_predictions
from egav.utils import ensure_dir, set_seed


def _maybe_add_use_mps_device(args_dict):
    try:
        import torch
        from transformers import TrainingArguments

        if not torch.backends.mps.is_available():
            return args_dict
        sig = inspect.signature(TrainingArguments.__init__)
        if "use_mps_device" in sig.parameters:
            args_dict["use_mps_device"] = True
        return args_dict
    except Exception:
        return args_dict


def postprocess_qa_predictions(
    examples,
    features,
    raw_predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int,
    max_answer_length: int,
    tokenizer,
):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = {}
    for i, feature in enumerate(features):
        example_id = feature["example_id"]
        example_index = example_id_to_index[example_id]
        features_per_example.setdefault(example_index, []).append(i)

    predictions = {}
    nbest_json = {}

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example.get(example_index, [])
        min_null_score = None
        prelim_predictions = []
        context = example["context"]

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or feature_null_score < min_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-n_best_size:][::-1]
            end_indexes = np.argsort(end_logits)[-n_best_size:][::-1]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    start_char, _ = offset_mapping[start_index]
                    _, end_char = offset_mapping[end_index]
                    text = context[start_char:end_char]
                    score = start_logits[start_index] + end_logits[end_index]
                    prelim_predictions.append(
                        {
                            "text": text,
                            "start_char": start_char,
                            "end_char": end_char,
                            "score": float(score),
                            "start_logit": float(start_logits[start_index]),
                            "end_logit": float(end_logits[end_index]),
                        }
                    )

        if not prelim_predictions:
            predictions[example["id"]] = ""
            nbest_json[example["id"]] = []
            continue

        prelim_predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)
        nbest = prelim_predictions[:n_best_size]
        scores = np.array([entry["score"] for entry in nbest])
        probs = np.exp(scores - np.max(scores))
        probs = probs / probs.sum()
        for idx, entry in enumerate(nbest):
            entry["probability"] = float(probs[idx])
            entry["rank"] = idx
        best = nbest[0]
        predictions[example["id"]] = best["text"]
        nbest_json[example["id"]] = nbest

    return predictions, nbest_json


def train_baseline(cfg):
    from datasets import load_dataset
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments

    set_seed(cfg.training.seed)
    raw_datasets = load_mlqa(cfg.data.dataset_name, cfg.data.languages, cfg.data.cache_dir)
    
    # MLQA doesn't have a train split - use SQuAD for training (standard approach)
    if cfg.data.train_split not in raw_datasets:
        print(f"Train split '{cfg.data.train_split}' not found in MLQA. Using SQuAD for training.")
        squad = load_dataset("rajpurkar/squad", split="train")
        # Add language field to match MLQA format
        squad = squad.map(lambda x: {"language": "en"})
        train_data = squad
    else:
        train_data = raw_datasets[cfg.data.train_split]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.qa_model_name, use_fast=True)
    train_dataset = train_data.map(
        lambda x: prepare_train_features(x, tokenizer, cfg.data.max_length, cfg.data.doc_stride),
        batched=True,
        remove_columns=train_data.column_names,
    )

    eval_dataset = raw_datasets[cfg.data.eval_split].map(
        lambda x: prepare_validation_features(x, tokenizer, cfg.data.max_length, cfg.data.doc_stride),
        batched=True,
        remove_columns=raw_datasets[cfg.data.eval_split].column_names,
    )

    output_dir = cfg.paths.baseline_dir / f"seed_{cfg.training.seed}"
    ensure_dir(output_dir)

    args_dict = dict(
        output_dir=str(output_dir),
        eval_strategy="steps",  # renamed from evaluation_strategy in newer transformers
        eval_steps=cfg.training.eval_steps,
        save_steps=cfg.training.save_steps,
        logging_steps=cfg.training.logging_steps,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        num_train_epochs=cfg.training.num_train_epochs,
        warmup_ratio=cfg.training.warmup_ratio,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        report_to=[],
        fp16=False,
        bf16=False,
    )
    args_dict = _maybe_add_use_mps_device(args_dict)
    training_args = TrainingArguments(**args_dict)

    model = AutoModelForQuestionAnswering.from_pretrained(cfg.model.qa_model_name)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    predictions = trainer.predict(eval_dataset)
    preds, nbest = postprocess_qa_predictions(
        raw_datasets[cfg.data.eval_split],
        eval_dataset,
        predictions.predictions,
        cfg.data.n_best_size,
        cfg.data.max_answer_length,
        tokenizer,
    )

    metrics = compute_metrics_from_predictions(preds, raw_datasets[cfg.data.eval_split])
    with (output_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (output_dir / "predictions_dev.json").open("w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=True)
    with (output_dir / "nbest_dev.json").open("w", encoding="utf-8") as f:
        json.dump(nbest, f, indent=2, ensure_ascii=True)
    with (output_dir / "predictions_dev.jsonl").open("w", encoding="utf-8") as f:
        for ex in raw_datasets[cfg.data.eval_split]:
            nbest_list = nbest.get(ex["id"], [])
            best = nbest_list[0] if nbest_list else {"text": "", "start_char": -1, "end_char": -1, "score": 0.0}
            row = {
                "id": ex["id"],
                "pred_text": best["text"],
                "pred_start": best["start_char"],
                "pred_end": best["end_char"],
                "score": best["score"],
                "n_best": nbest_list,
            }
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", type=str, default="en")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = default_config()
    cfg.data.languages = args.languages.split(",")
    cfg.training.seed = args.seed
    metrics = train_baseline(cfg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
