import json
import os
import random
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve


# ---------------------------------------------------------------------------
# Hub dataset identifiers for MLQA
# ---------------------------------------------------------------------------
# The canonical MLQA dataset on Hugging Face is "facebook/mlqa".
# Config names follow the pattern "mlqa.{context_lang}.{question_lang}".
# Supported languages: ar, de, en, es, hi, vi, zh
#
# Because recent versions of `datasets` (>=4.x) dropped support for dataset
# scripts entirely, we fall back to downloading the raw MLQA JSON files from
# the official GitHub release and loading them directly.
# ---------------------------------------------------------------------------
_MLQA_HUB_NAME = "facebook/mlqa"
_MLQA_LANGUAGES = {"ar", "de", "en", "es", "hi", "vi", "zh"}

# Official MLQA download URLs (from GitHub release)
_MLQA_DEV_URL = "https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip"


def _mlqa_config_name(lang: str) -> str:
    """Return the MLQA Hub config name for a language (context==question language)."""
    return f"mlqa.{lang}.{lang}"


def _get_mlqa_cache_dir(cache_dir: Optional[str]) -> Path:
    """Return the directory where MLQA data will be cached."""
    if cache_dir:
        return Path(cache_dir) / "mlqa_raw"
    return Path.home() / ".cache" / "egav" / "mlqa_raw"


def _download_mlqa_if_needed(cache_dir: Optional[str]) -> Path:
    """Download and extract MLQA dataset if not already cached.
    
    Returns the path to the extracted MLQA_V1 directory.
    """
    mlqa_cache = _get_mlqa_cache_dir(cache_dir)
    mlqa_cache.mkdir(parents=True, exist_ok=True)
    
    mlqa_dir = mlqa_cache / "MLQA_V1"
    if mlqa_dir.exists() and (mlqa_dir / "dev").exists():
        return mlqa_dir
    
    zip_path = mlqa_cache / "MLQA_V1.zip"
    if not zip_path.exists():
        print(f"Downloading MLQA dataset to {zip_path}...")
        urlretrieve(_MLQA_DEV_URL, zip_path)
        print("Download complete.")
    
    print(f"Extracting MLQA dataset to {mlqa_cache}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(mlqa_cache)
    print("Extraction complete.")
    
    return mlqa_dir


def _load_mlqa_from_raw_json(lang: str, cache_dir: Optional[str]):
    """Load MLQA from the official raw JSON files.
    
    MLQA structure after extraction:
        MLQA_V1/
            dev/dev-context-{lang}-question-{lang}.json
            test/test-context-{lang}-question-{lang}.json
    
    Note: MLQA does NOT have an official train split. For training, users
    typically use SQuAD or translate SQuAD to other languages.
    """
    from datasets import DatasetDict
    
    mlqa_dir = _download_mlqa_if_needed(cache_dir)
    
    dd = DatasetDict()
    
    # Dev split -> validation
    dev_file = mlqa_dir / "dev" / f"dev-context-{lang}-question-{lang}.json"
    if dev_file.exists():
        dd["validation"] = _load_squad_like_json(dev_file, language=lang)
    
    # Test split
    test_file = mlqa_dir / "test" / f"test-context-{lang}-question-{lang}.json"
    if test_file.exists():
        dd["test"] = _load_squad_like_json(test_file, language=lang)
    
    if not dd:
        raise FileNotFoundError(
            f"Could not find MLQA files for language '{lang}' in {mlqa_dir}. "
            f"Expected files like dev-context-{lang}-question-{lang}.json"
        )
    
    return dd


def _load_dataset_from_hub(name: str, config_name: str, cache_dir: Optional[str]):
    """Load a dataset from the Hugging Face Hub.

    To avoid `datasets` accidentally resolving to a local `mlqa.py` file (which
    newer versions refuse to execute), we temporarily:
      1. Change CWD to a temp directory
      2. Remove CWD and common problematic paths from sys.path
      3. Rename any local mlqa.py temporarily (belt-and-suspenders)

    This ensures `datasets` only looks at the Hub.
    """
    from datasets import load_dataset

    old_cwd = os.getcwd()
    old_sys_path = list(sys.path)

    # Check for local mlqa.py in CWD and temporarily rename it
    local_mlqa = Path(old_cwd) / "mlqa.py"
    renamed_mlqa = None
    if local_mlqa.exists():
        renamed_mlqa = local_mlqa.with_suffix(".py.bak_egav")
        try:
            local_mlqa.rename(renamed_mlqa)
        except Exception:
            renamed_mlqa = None  # couldn't rename, proceed anyway

    try:
        with tempfile.TemporaryDirectory(prefix="egav_hf_") as tmp:
            os.chdir(tmp)
            # Remove entries that might contain local scripts
            sys.path = [
                p for p in sys.path
                if p not in {"", ".", old_cwd}
                and not (p and Path(p).resolve() == Path(old_cwd).resolve())
            ]

            # Try without trust_remote_code first (for newer datasets versions)
            try:
                return load_dataset(name, config_name, cache_dir=cache_dir)
            except (TypeError, RuntimeError):
                pass
            
            # Try with trust_remote_code=True (for older datasets versions)
            try:
                return load_dataset(name, config_name, cache_dir=cache_dir, trust_remote_code=True)
            except (TypeError, RuntimeError):
                pass
            
            # If both fail, raise to trigger fallback
            raise RuntimeError("Hub loading failed")
    finally:
        os.chdir(old_cwd)
        sys.path = old_sys_path
        # Restore the renamed file
        if renamed_mlqa and renamed_mlqa.exists():
            try:
                renamed_mlqa.rename(local_mlqa)
            except Exception:
                pass


def _normalize_dataset_name(dataset_name: str) -> str:
    """Normalize dataset identifiers.

    Newer versions of `datasets` disallow loading *local* dataset scripts (e.g., mlqa.py).
    Users sometimes download `mlqa.py` and pass it as dataset name; map that to the Hub
    dataset id instead.
    """

    name = (dataset_name or "").strip()
    if not name:
        return "mlqa"

    p = Path(name)
    if p.suffix == ".py":
        warnings.warn(
            f"dataset_name={dataset_name!r} looks like a local dataset script. "
            "Local dataset scripts are not supported by recent `datasets`; "
            "falling back to the Hub dataset id 'mlqa'.",
            RuntimeWarning,
        )
        return p.stem
    return name


def _load_squad_like_json(path: Path, language: Optional[str] = None):
    """Load a SQuAD/MLQA-style JSON file into a HF Dataset."""
    from datasets import Dataset

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    data = obj.get("data", obj)
    rows = []
    for article in data:
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                answers = qa.get("answers", []) or []
                rows.append(
                    {
                        "id": qa.get("id", ""),
                        "question": qa.get("question", ""),
                        "context": context,
                        "answers": {
                            "text": [a.get("text", "") for a in answers],
                            "answer_start": [a.get("answer_start", -1) for a in answers],
                        },
                        "language": language or "",
                    }
                )
    return Dataset.from_list(rows)


def _try_load_local_mlqa(dataset_path: Path, lang: str):
    """Attempt to load MLQA from local JSON files.

    Supports:
    - a directory containing split JSON files, optionally per-language
    - a single JSON file (treated as a single split)
    """
    from datasets import DatasetDict

    if dataset_path.is_file() and dataset_path.suffix.lower() == ".json":
        return DatasetDict({"validation": _load_squad_like_json(dataset_path, language=lang)})

    if not dataset_path.is_dir():
        raise ValueError(f"Unsupported local dataset path: {dataset_path}")

    # Try common layouts:
    #   root/{lang}/{train,dev,test}.json
    #   root/{train,dev,test}_{lang}.json
    #   root/{train,dev,test}.json
    candidates = []
    lang_dir = dataset_path / lang
    if lang_dir.is_dir():
        candidates.append(lang_dir)
    candidates.append(dataset_path)

    split_map = {
        "train": "train",
        "training": "train",
        "dev": "validation",
        "valid": "validation",
        "validation": "validation",
        "val": "validation",
        "test": "test",
    }

    dd = DatasetDict()
    for base in candidates:
        for file in base.glob("*.json"):
            lower = file.stem.lower()
            # e.g. dev_en, en_dev, etc.
            parts = [p for p in lower.replace("-", "_").split("_") if p]
            split_key = None
            for p in parts:
                if p in split_map:
                    split_key = split_map[p]
                    break
            if split_key is None:
                continue
            if split_key in dd:
                continue
            dd[split_key] = _load_squad_like_json(file, language=lang)

        if dd:
            return dd
    raise FileNotFoundError(
        f"Could not find MLQA-style JSON splits under {dataset_path} (lang={lang})."
    )


def _try_load_mlqa(dataset_name: str, lang: str, cache_dir: Optional[str]):
    name = _normalize_dataset_name(dataset_name)
    p = Path(dataset_name)

    # If the user passed a local path, try local JSON loading.
    if p.exists() and p.suffix != ".py":
        return _try_load_local_mlqa(p, lang)

    # Map short names like "mlqa" to the canonical Hub path "facebook/mlqa".
    if name.lower() in {"mlqa", "mlqa.py"}:
        name = _MLQA_HUB_NAME

    # For MLQA, config names are "mlqa.{context_lang}.{question_lang}".
    if name == _MLQA_HUB_NAME:
        if lang not in _MLQA_LANGUAGES:
            raise ValueError(
                f"Language '{lang}' is not supported by MLQA. "
                f"Supported languages: {sorted(_MLQA_LANGUAGES)}"
            )
        config_name = _mlqa_config_name(lang)
        
        # Try Hub first, fall back to raw JSON download if Hub fails
        # (required for datasets>=4.x which dropped dataset script support)
        try:
            return _load_dataset_from_hub(name, config_name, cache_dir)
        except Exception as e:
            warnings.warn(
                f"Could not load MLQA from Hub ({e}). "
                "Falling back to downloading raw JSON files from Facebook.",
                RuntimeWarning,
            )
            return _load_mlqa_from_raw_json(lang, cache_dir)
    else:
        config_name = lang
        # For non-MLQA datasets, just try the Hub
        return _load_dataset_from_hub(name, config_name, cache_dir)


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
