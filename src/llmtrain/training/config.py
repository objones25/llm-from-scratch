from dataclasses import dataclass


@dataclass
class DataConfig:
    dataset_name: str = "fineweb_edu"
    shuffle_buffer_size: int = 1000
    max_seq_len: int = 2048
    tokenizer_vocab_size: int = 32768
    tokenizer_sample_size: int = 200


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    dropout: float = 0.0
    rope_theta: float = 10000.0


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    max_steps: int = 10000
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000
    compile: bool = True
    use_amp: bool = True
    wandb_project: str = "llm-training"
    wandb_mode: str = "online"
    log_file: str = "app.log"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 50
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
