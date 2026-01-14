import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx

from egav.ner import Mention


@dataclass
class SentenceSpan:
    text: str
    start_char: int
    end_char: int


@dataclass
class Node:
    node_id: int
    node_type: str
    text: str
    start_char: int
    end_char: int
    sent_id: int


EDGE_TYPES = {
    "qent_span": 0,
    "qent_sent": 1,
    "cent_sent": 2,
    "span_cent": 3,
    "cent_cooccur": 4,
    "qent_sent_fallback": 5,
}


def split_sentences(text: str) -> List[SentenceSpan]:
    spans = []
    start = 0
    for match in re.finditer(r"[.!?]+\s+", text):
        end = match.end()
        spans.append(SentenceSpan(text=text[start:end].strip(), start_char=start, end_char=end))
        start = end
    if start < len(text):
        spans.append(SentenceSpan(text=text[start:].strip(), start_char=start, end_char=len(text)))
    return [s for s in spans if s.text]


def assign_sentence_ids(mentions: List[Mention], sentences: List[SentenceSpan]) -> Dict[int, int]:
    mapping = {}
    for idx, m in enumerate(mentions):
        sent_id = -1
        for s_idx, s in enumerate(sentences):
            if m.start_char >= s.start_char and m.end_char <= s.end_char:
                sent_id = s_idx
                break
        mapping[idx] = sent_id
    return mapping


def _normalized_contains(sentence: str, phrase: str) -> bool:
    norm_sent = re.sub(r"\s+", " ", sentence.lower())
    norm_phrase = re.sub(r"\s+", " ", phrase.lower())
    return norm_phrase in norm_sent


def build_example_graph(
    question: str,
    context: str,
    q_mentions: List[Mention],
    c_mentions: List[Mention],
    candidates: List[Dict],
    sentences: Optional[List[SentenceSpan]] = None,
) -> Tuple[nx.DiGraph, List[Node], List[SentenceSpan]]:
    if sentences is None:
        sentences = split_sentences(context)

    graph = nx.DiGraph()
    nodes: List[Node] = []

    node_id = 0
    qent_nodes = []
    for m in q_mentions:
        nodes.append(Node(node_id, "QENT", m.text, m.start_char, m.end_char, -1))
        qent_nodes.append(node_id)
        node_id += 1

    cent_nodes = []
    cent_sent_map = assign_sentence_ids(c_mentions, sentences)
    for idx, m in enumerate(c_mentions):
        nodes.append(Node(node_id, "CENT", m.text, m.start_char, m.end_char, cent_sent_map[idx]))
        cent_nodes.append(node_id)
        node_id += 1

    span_nodes = []
    for cand in candidates:
        nodes.append(Node(node_id, "SPAN", cand["span_text"], cand["start_char"], cand["end_char"], -1))
        span_nodes.append(node_id)
        node_id += 1

    sent_nodes = []
    for s in sentences:
        nodes.append(Node(node_id, "SENT", s.text, s.start_char, s.end_char, -1))
        sent_nodes.append(node_id)
        node_id += 1

    for n in nodes:
        graph.add_node(n.node_id, node_type=n.node_type)

    # QENT -> SPAN
    for q_id in qent_nodes:
        for s_id in span_nodes:
            graph.add_edge(q_id, s_id, edge_type="qent_span")

    # CENT -> SENT
    for c_id in cent_nodes:
        sent_id = nodes[c_id].sent_id
        if sent_id >= 0:
            s_node_id = sent_nodes[sent_id]
            graph.add_edge(c_id, s_node_id, edge_type="cent_sent")

    # QENT -> SENT (match or fallback)
    for q_idx, q_id in enumerate(qent_nodes):
        matched = False
        q_text = q_mentions[q_idx].text
        for s_idx, s_node_id in enumerate(sent_nodes):
            if _normalized_contains(sentences[s_idx].text, q_text):
                graph.add_edge(q_id, s_node_id, edge_type="qent_sent")
                matched = True
        if not matched:
            for s_node_id in sent_nodes:
                graph.add_edge(q_id, s_node_id, edge_type="qent_sent_fallback")

    # SPAN -> CENT (overlap or same sentence)
    for span_idx, span_id in enumerate(span_nodes):
        span = nodes[span_id]
        span_sent_id = -1
        for s_idx, s in enumerate(sentences):
            if span.start_char >= s.start_char and span.end_char <= s.end_char:
                span_sent_id = s_idx
                break
        for c_idx, c_id in enumerate(cent_nodes):
            cent = nodes[c_id]
            overlaps = cent.start_char >= span.start_char and cent.end_char <= span.end_char
            same_sent = span_sent_id >= 0 and cent.sent_id == span_sent_id
            if overlaps or same_sent:
                graph.add_edge(span_id, c_id, edge_type="span_cent")

    # CENT <-> CENT co-occur
    for i, c_id in enumerate(cent_nodes):
        for j in range(i + 1, len(cent_nodes)):
            other_id = cent_nodes[j]
            if nodes[c_id].sent_id >= 0 and nodes[c_id].sent_id == nodes[other_id].sent_id:
                graph.add_edge(c_id, other_id, edge_type="cent_cooccur")
                graph.add_edge(other_id, c_id, edge_type="cent_cooccur")

    return graph, nodes, sentences


def graph_to_tensors(graph: nx.DiGraph, nodes: List[Node]):
    type_to_idx = {"QENT": 0, "CENT": 1, "SPAN": 2, "SENT": 3}
    node_type = []
    node_features = []
    span_node_ids = []
    for node in nodes:
        t_idx = type_to_idx[node.node_type]
        node_type.append(t_idx)
        type_onehot = [0.0] * len(type_to_idx)
        type_onehot[t_idx] = 1.0
        length_feat = min(len(node.text) / 50.0, 1.0)
        node_features.append(type_onehot + [length_feat])
        if node.node_type == "SPAN":
            span_node_ids.append(node.node_id)

    edge_index = []
    edge_type = []
    for u, v, attrs in graph.edges(data=True):
        edge_index.append([u, v])
        edge_type.append(EDGE_TYPES.get(attrs.get("edge_type"), 0))

    return {
        "node_features": node_features,
        "node_type": node_type,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "span_node_ids": span_node_ids,
    }
