import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Mention:
    text: str
    start_char: int
    end_char: int
    label: str
    norm: str


_NER_CACHE: Dict[Tuple[str, str], object] = {}


def normalize_mention(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^[\W_]+|[\W_]+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"\d{1,4}", "<NUM>", cleaned)
    return cleaned


def get_hf_ner_pipeline(model_name: str):
    from transformers import pipeline

    return pipeline("token-classification", model=model_name, aggregation_strategy="simple")


def extract_mentions(text: str, lang: str = "en", model_name: str = "Davlan/xlm-roberta-base-ner-hrl") -> List[Mention]:
    key = (lang, model_name)
    if key not in _NER_CACHE:
        _NER_CACHE[key] = get_hf_ner_pipeline(model_name)
    ner = _NER_CACHE[key]
    results = ner(text)
    mentions: List[Mention] = []
    for r in results:
        mention_text = text[r["start"] : r["end"]]
        mentions.append(
            Mention(
                text=mention_text,
                start_char=int(r["start"]),
                end_char=int(r["end"]),
                label=r.get("entity_group", r.get("entity", "MISC")),
                norm=normalize_mention(mention_text),
            )
        )
    return mentions


def mentions_in_span(mentions: List[Mention], start: int, end: int) -> List[Mention]:
    return [m for m in mentions if m.start_char >= start and m.end_char <= end]
