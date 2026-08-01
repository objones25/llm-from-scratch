from llmtrain.training.config import DataConfig, GenerationConfig, ModelConfig, TrainConfig


def test_data_config_has_sensible_defaults():
    cfg = DataConfig()
    assert cfg.dataset_name == "fineweb_edu"
    assert cfg.shuffle_buffer_size > 0
    assert cfg.max_seq_len > 0
    assert cfg.tokenizer_vocab_size > 0
    assert cfg.tokenizer_sample_size > 0


def test_model_config_is_overridable():
    cfg = ModelConfig(d_model=64, n_layers=4)
    assert cfg.d_model == 64
    assert cfg.n_layers == 4


def test_model_config_has_rope_and_gqa_defaults():
    cfg = ModelConfig()
    assert cfg.rope_theta == 10000.0
    assert cfg.n_kv_heads == 4


def test_train_config_has_sensible_defaults():
    cfg = TrainConfig()
    assert cfg.max_steps > 0
    assert cfg.batch_size > 0
    assert cfg.checkpoint_interval > 0
    assert cfg.weight_decay >= 0
    assert 0 < cfg.beta1 < 1
    assert 0 < cfg.beta2 < 1


def test_generation_config_has_sensible_defaults():
    cfg = GenerationConfig()
    assert cfg.max_new_tokens > 0
    assert cfg.temperature > 0
    assert cfg.repetition_penalty >= 1.0
    assert cfg.top_k >= 0
    assert cfg.top_p > 0
