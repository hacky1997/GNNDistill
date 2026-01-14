import random
from typing import Dict, List, Optional, Tuple


def _try_load_mlqa(dataset_name: str, lang: str, cache_dir: Optional[str]):
    from datasets import load_dataset

    try:
        return load_dataset(dataset_name, lang, cache_dir=cache_dir)
    except Exception:
        return load_dataset(dataset_name, name=lang, cache_dir=cache_dir)


def load_mlqa(dataset_name: str = "mlqa", languages: Optional[List[str]] = None, cache_dir: Optional[str] = None):
    from datasets import concatenate_datasets

    if not languages:
        languages = ["en"]
    dataset_splits = {}
    for lang in languages:
        ds = _try_load_mlqa(dataset_name, lang, cache_dir)
        for split, split_ds in ds.items():
            if split not in dataset_splits:
                dataset_splits[split] = [split_ds]
            else:
                dataset_splits[split].append(split_ds)
    merged = {}
    for split, parts in dataset_splits.items():
        if len(parts) == 1:
            merged[split] = parts[0]
        else:
            merged[split] = concatenate_datasets(parts)
    return merged


def prepare_train_features(examples, tokenizer, max_length: int, doc_stride: int):
    questions = [q.lstrip() for q in examples["question"]]
    contexts = examples["context"]
    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            start_positions.append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def prepare_validation_features(examples, tokenizer, max_length: int, doc_stride: int):
    questions = [q.lstrip() for q in examples["question"]]
    contexts = examples["context"]
    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["example_id"] = []
    for i in range(len(tokenized["input_ids"])):
        sample_index = sample_mapping[i]
        tokenized["example_id"].append(examples["id"][sample_index])
        sequence_ids = tokenized.sequence_ids(i)
        offset_mapping = tokenized["offset_mapping"][i]
        tokenized["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None) for k, o in enumerate(offset_mapping)
        ]
    return tokenized


def verify_gold_offsets(dataset, num_samples: int = 50) -> List[Tuple[str, bool]]:
    results = []
    total = len(dataset)
    indices = random.sample(range(total), min(num_samples, total))
    for idx in indices:
        ex = dataset[idx]
        answers = ex.get("answers", {})
        if not answers or not answers.get("text"):
            results.append((ex["id"], False))
            continue
        answer_text = answers["text"][0]
        start = answers["answer_start"][0]
        context = ex["context"]
        ok = context[start : start + len(answer_text)] == answer_text
        results.append((ex["id"], ok))
    return results


def get_collate_fn(tokenizer):
    from transformers import default_data_collator

    return default_data_collator
