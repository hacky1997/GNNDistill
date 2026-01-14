import math
import re
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from egav.graph_build import build_example_graph, split_sentences
from egav.ner import Mention, mentions_in_span
from egav.utils import entropy


def infer_answer_type(question: str) -> str:
    q = question.strip().lower()
    if q.startswith("who") or q.startswith("whom"):
        return "PERSON"
    if q.startswith("when") or "year" in q or "date" in q:
        return "DATE"
    if q.startswith("where") or "which city" in q or "which country" in q:
        return "GPE"
    if q.startswith("how many") or q.startswith("how much"):
        return "NUMBER"
    if q.startswith("which organization") or q.startswith("which organisation"):
        return "ORG"
    return "UNKNOWN"


def label_matches(expected: str, label: str) -> bool:
    if expected == "UNKNOWN":
        return False
    label = label.upper()
    mapping = {
        "PERSON": {"PER", "PERSON"},
        "DATE": {"DATE", "TIME"},
        "GPE": {"LOC", "GPE", "LOCATION"},
        "NUMBER": {"CARDINAL", "QUANTITY", "NUM", "NUMBER", "MONEY"},
        "ORG": {"ORG", "ORGANIZATION"},
    }
    return label in mapping.get(expected, set())


def get_numbers(text: str) -> List[str]:
    return re.findall(r"\b\d+(?:[\.,]\d+)?\b", text)


def has_negation(text: str) -> bool:
    neg_terms = ["not", "no", "never", "neither", "instead", "however", "but"]
    lowered = text.lower()
    return any(term in lowered for term in neg_terms)


def entity_signature(mentions: List[Mention]) -> Set[str]:
    return {m.norm for m in mentions if m.norm}


def feature_names() -> List[str]:
    return [
        "shortest_path_len",
        "supporting_sent_count",
        "bridge_entity_count",
        "answer_type_match",
        "numeric_match",
        "numeric_mismatch",
        "negation_in_sentence",
        "span_logit_sum",
        "start_logit",
        "end_logit",
        "span_prob",
        "span_rank",
        "margin_top12",
        "nbest_entropy",
        "span_char_len",
        "span_token_len",
        "entity_sig_size",
        "entity_overlap_top1",
        "entity_overlap_top2",
        "disjoint_top2",
        "qent_overlap_count",
        "span_is_top1",
        "q_numbers_count",
        "span_numbers_count",
    ]


def compute_features_for_candidates(
    question: str,
    context: str,
    candidates: List[Dict],
    q_mentions: List[Mention],
    c_mentions: List[Mention],
    sentences=None,
):
    if sentences is None:
        sentences = split_sentences(context)
    graph, nodes, sentences = build_example_graph(question, context, q_mentions, c_mentions, candidates, sentences)

    qent_node_ids = [n.node_id for n in nodes if n.node_type == "QENT"]
    cent_node_ids = [n.node_id for n in nodes if n.node_type == "CENT"]
    span_node_ids = [n.node_id for n in nodes if n.node_type == "SPAN"]

    q_sent_ids = set()
    for s_idx, s in enumerate(sentences):
        for q_m in q_mentions:
            if q_m.norm and q_m.norm in s.text.lower():
                q_sent_ids.add(s_idx)

    mention_sent_ids = []
    for m in c_mentions:
        m_sent_id = -1
        for s_idx, s in enumerate(sentences):
            if m.start_char >= s.start_char and m.end_char <= s.end_char:
                m_sent_id = s_idx
                break
        mention_sent_ids.append(m_sent_id)

    span_entity_sets = []
    span_sentence_ids = []
    for cand in candidates:
        s_id = -1
        for idx, s in enumerate(sentences):
            if cand["start_char"] >= s.start_char and cand["end_char"] <= s.end_char:
                s_id = idx
                break
        span_sentence_ids.append(s_id)
        span_mentions = mentions_in_span(c_mentions, cand["start_char"], cand["end_char"])
        if not span_mentions and s_id >= 0:
            span_mentions = [m for m in c_mentions if m.start_char >= sentences[s_id].start_char and m.end_char <= sentences[s_id].end_char]
        span_entity_sets.append(entity_signature(span_mentions))

    sorted_indices = sorted(range(len(candidates)), key=lambda i: candidates[i].get("span_logit_sum", 0.0), reverse=True)
    top1_idx = sorted_indices[0] if sorted_indices else None
    top2_idx = sorted_indices[1] if len(sorted_indices) > 1 else None
    top1_set = span_entity_sets[top1_idx] if top1_idx is not None else set()
    top2_set = span_entity_sets[top2_idx] if top2_idx is not None else set()

    probs = [c.get("span_prob", 0.0) for c in candidates]
    entropy_val = entropy(probs) if probs else 0.0
    margin_top12 = 0.0
    if len(sorted_indices) > 1:
        margin_top12 = candidates[sorted_indices[0]]["span_logit_sum"] - candidates[sorted_indices[1]]["span_logit_sum"]

    expected_type = infer_answer_type(question)
    q_numbers = get_numbers(question)

    features = []
    for idx, cand in enumerate(candidates):
        span_nodes = [span_node_ids[idx]]
        span_cent_nodes = []
        for span_id in span_nodes:
            for _, v, attrs in graph.out_edges(span_id, data=True):
                if attrs.get("edge_type") == "span_cent":
                    span_cent_nodes.append(v)

        shortest = 999.0
        bridge_entities = 0.0
        if span_cent_nodes and qent_node_ids:
            for q_id in qent_node_ids:
                for c_id in span_cent_nodes:
                    try:
                        path = nx.shortest_path(graph, q_id, c_id)
                        length = len(path) - 1
                        if length < shortest:
                            shortest = float(length)
                            bridge_entities = float(
                                sum(1 for n_id in path if nodes[n_id].node_type == "CENT")
                            )
                    except nx.NetworkXNoPath:
                        continue

        support_sent_count = 0
        span_sent_id = span_sentence_ids[idx]
        span_mention_indices = [
            i for i, m in enumerate(c_mentions) if m.start_char >= cand["start_char"] and m.end_char <= cand["end_char"]
        ]
        span_sent_ids = {mention_sent_ids[i] for i in span_mention_indices if mention_sent_ids[i] >= 0}
        if span_sent_ids:
            support_sent_count = len(span_sent_ids & q_sent_ids)
        elif span_sent_id >= 0 and span_sent_id in q_sent_ids:
            support_sent_count = 1

        span_mentions = mentions_in_span(c_mentions, cand["start_char"], cand["end_char"])
        answer_type_match = 1.0 if any(label_matches(expected_type, m.label) for m in span_mentions) else 0.0

        span_numbers = get_numbers(cand["span_text"])
        numeric_match = 1.0 if q_numbers and any(num in span_numbers for num in q_numbers) else 0.0
        numeric_mismatch = 1.0 if q_numbers and not span_numbers else 0.0

        negation = 0.0
        if span_sent_id >= 0:
            negation = 1.0 if has_negation(sentences[span_sent_id].text) else 0.0

        span_set = span_entity_sets[idx]
        overlap_top1 = jaccard(span_set, top1_set)
        overlap_top2 = jaccard(span_set, top2_set)
        disjoint_top2 = 1.0 if overlap_top1 == 0.0 and overlap_top2 == 0.0 else 0.0

        qent_overlap = 1.0 if span_sent_id in q_sent_ids else 0.0
        span_is_top1 = 1.0 if top1_idx is not None and idx == top1_idx else 0.0

        features.append(
            [
                shortest,
                float(support_sent_count),
                bridge_entities,
                answer_type_match,
                numeric_match,
                numeric_mismatch,
                negation,
                float(cand.get("span_logit_sum", 0.0)),
                float(cand.get("start_logit", 0.0)),
                float(cand.get("end_logit", 0.0)),
                float(cand.get("span_prob", 0.0)),
                float(cand.get("rank", 0.0)),
                float(margin_top12),
                float(entropy_val),
                float(len(cand.get("span_text", ""))),
                float(len(cand.get("span_text", "").split())),
                float(len(span_set)),
                float(overlap_top1),
                float(overlap_top2),
                float(disjoint_top2),
                float(qent_overlap),
                float(span_is_top1),
                float(len(q_numbers)),
                float(len(span_numbers)),
            ]
        )

    return features


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
