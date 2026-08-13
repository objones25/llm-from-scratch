from dataclasses import dataclass


@dataclass
class DataConfig:
    dataset_name: str = "fineweb_edu"
    shuffle_buffer_size: int = 1000
    max_seq_len: int = 2048
    tokenizer_vocab_size: int = 32768
    # 200 was enough for the tiny_shakespeare smoke test but too thin a corpus to derive a
    # 32k-vocab BPE tokenizer that generalizes to the real fineweb_edu pretraining run —
    # bumped for a bigger future run; tiny_shakespeare (472 rows) just takes what's available.
    tokenizer_sample_size: int = 10000


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 1440
    n_layers: int = 20
    n_heads: int = 20
    n_kv_heads: int = 4
    dropout: float = 0.0
    rope_theta: float = 10000.0


@dataclass
class TrainConfig:
    batch_size: int = 32
    gradient_accumulation_steps: int = 8
    grad_clip: float = 1.0
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_steps: int = 18500
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 125
    keep_last_n_checkpoints: int = 3
    eval_interval: int = 500
    compile: bool = True
    use_amp: bool = True
    use_fused_ce: bool = True
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
