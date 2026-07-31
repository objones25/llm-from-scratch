from dataclasses import dataclass


@dataclass
class DataConfig:
    dataset_name: str = "tiny_shakespeare"
    shuffle_buffer_size: int = 1000
    max_seq_len: int = 128
    tokenizer_vocab_size: int = 1000


@dataclass
class ModelConfig:
    vocab_size: int = 1000
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 1
    max_seq_len: int = 128
    dropout: float = 0.0
    rope_theta: float = 10000.0


@dataclass
class TrainConfig:
    batch_size: int = 8
    lr: float = 3e-4
    max_steps: int = 100
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 50
    compile: bool = True
    use_amp: bool = True
    wandb_project: str = "llm-training"
    wandb_mode: str = "online"
    log_file: str = "app.log"
