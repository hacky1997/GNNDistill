EGAV: Entity-Graph Verification + Reranking + Abstention + Interpretable Ambiguity

This repo implements the EGAV pipeline end-to-end with a baseline QA model, candidate generation, entity graph features, a verifier (MLP baseline), abstention/clarify policy, evaluation, and visualization.

Quick start (Mac M1/M2)

1) Install deps

```
pip install -r requirements.txt
```

2) Train baseline QA (XLM-R)

```
python -m egav.qa_baseline --languages en --seed 42
```

3) Dump candidates (top-K spans)

```
python -m egav.candidates \
  --model runs/baseline/seed_42 \
  --output runs/baseline/candidates_dev.jsonl \
  --split validation \
  --languages en
```

4) Train verifier (MLP)

```
python -m egav.train_verifier \
  --candidates runs/baseline/candidates_dev.jsonl \
  --output runs/verifier/seed_42 \
  --lang en
```

5) Rerank + abstention

```
python -m egav.inference \
  --candidates runs/baseline/candidates_dev.jsonl \
  --verifier runs/verifier/seed_42/verifier_mlp.pt \
  --output runs/results/preds_dev.jsonl \
  --gamma 0.5 --tau_correct 0.5 --tau_margin 0.1
```

Notes
- MPS is auto-selected when available (Mac M1/M2). The QA trainer disables fp16/bf16 for MPS stability.
- MLQA configs vary; if your split names differ, update `egav/config.py`.
- GNN verifier is optional and requires `torch-geometric` (not MPS-friendly yet).

Key modules
- `egav/data_mlqa.py`: dataset loading and tokenization
- `egav/qa_baseline.py`: baseline QA fine-tuning
- `egav/candidates.py`: top-K span extraction
- `egav/ner.py`: multilingual NER
- `egav/graph_build.py`: per-example graph construction
- `egav/graph_features.py`: interpretable features
- `egav/train_verifier.py`: MLP verifier training
- `egav/inference.py`: safe reranking + abstention
- `egav/evaluation.py`: EM/F1, coverage, AURC
- `egav/visualize.py`: paper-ready plots
