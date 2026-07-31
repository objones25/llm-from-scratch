from llmtrain.training.config import DataConfig, ModelConfig, TrainConfig


def test_data_config_has_sensible_defaults():
    cfg = DataConfig()
    assert cfg.dataset_name == "tiny_shakespeare"
    assert cfg.shuffle_buffer_size > 0
    assert cfg.max_seq_len > 0
    assert cfg.tokenizer_vocab_size > 0


def test_model_config_is_overridable():
    cfg = ModelConfig(d_model=64, n_layers=4)
    assert cfg.d_model == 64
    assert cfg.n_layers == 4


def test_train_config_has_sensible_defaults():
    cfg = TrainConfig()
    assert cfg.max_steps > 0
    assert cfg.batch_size > 0
    assert cfg.checkpoint_interval > 0
