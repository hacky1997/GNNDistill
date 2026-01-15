from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PathsConfig:
    repo_root: Path = Path(__file__).resolve().parents[1]
    runs_dir: Path = Path(__file__).resolve().parents[1] / "runs"
    baseline_dir: Path = Path(__file__).resolve().parents[1] / "runs" / "baseline"
    verifier_dir: Path = Path(__file__).resolve().parents[1] / "runs" / "verifier"
    results_dir: Path = Path(__file__).resolve().parents[1] / "runs" / "results"


@dataclass
class DataConfig:
    dataset_name: str = "mlqa"
    languages: List[str] = field(default_factory=lambda: ["en"])
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: str = "test"
    max_length: int = 384
    doc_stride: int = 128
    max_answer_length: int = 30
    n_best_size: int = 20
    cache_dir: Optional[str] = None


@dataclass
class ModelConfig:
    qa_model_name: str = "xlm-roberta-base"
    ner_model_name: str = "Davlan/xlm-roberta-base-ner-hrl"
    gnn_hidden_dim: int = 128


@dataclass
class TrainingConfig:
    seed: int = 42
    num_train_epochs: int = 200
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50
    max_grad_norm: float = 1.0


@dataclass
class VerifierConfig:
    feature_dim: int = 24
    hidden_dim: int = 128
    dropout: float = 0.2
    lambda_rank: float = 1.0
    rank_margin: float = 0.05
    huber_delta: float = 0.1
    num_train_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128


@dataclass
class InferenceConfig:
    gamma: float = 0.5
    tau_correct: float = 0.5
    tau_margin: float = 0.1
    max_candidates: int = 20


@dataclass
class EvalConfig:
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 7


@dataclass
class PlotConfig:
    dpi: int = 300
    font_family: str = "Times New Roman"
    base_font_size: int = 12


@dataclass
class EGAVConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)


def default_config() -> EGAVConfig:
    return EGAVConfig()
