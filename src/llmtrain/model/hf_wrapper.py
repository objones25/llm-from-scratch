import torch
from tokenizers import Tokenizer
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from llmtrain.data.tokenizer import PAD_TOKEN, UNK_TOKEN
from llmtrain.model.transformer import TransformerLM
from llmtrain.training.config import ModelConfig


class TransformerLMConfig(PretrainedConfig):
    model_type = "transformer_lm"

    def __init__(
        self,
        vocab_size: int = 32768,
        d_model: int = 1440,
        n_layers: int = 20,
        n_heads: int = 20,
        n_kv_heads: int = 4,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.dropout = dropout
        self.rope_theta = rope_theta
        # Weight tying is already handled directly on the parameter in
        # TransformerLM.__init__ (self.head.weight = self.token_emb.weight) -- HF's own
        # tie-weights machinery should stay inert rather than fight it.
        kwargs["tie_word_embeddings"] = False
        super().__init__(**kwargs)

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            dropout=self.dropout,
            rope_theta=self.rope_theta,
        )

    @classmethod
    def from_model_config(cls, model_config: ModelConfig) -> "TransformerLMConfig":
        return cls(
            vocab_size=model_config.vocab_size,
            d_model=model_config.d_model,
            n_layers=model_config.n_layers,
            n_heads=model_config.n_heads,
            n_kv_heads=model_config.n_kv_heads,
            dropout=model_config.dropout,
            rope_theta=model_config.rope_theta,
        )


class TransformerLMForCausalLM(PreTrainedModel):
    config_class = TransformerLMConfig

    def __init__(self, config: TransformerLMConfig) -> None:
        super().__init__(config)
        self.model = TransformerLM(config.to_model_config())

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs
    ) -> CausalLMOutputWithPast:
        # attention_mask is accepted (required by HF's calling convention, and DPOTrainer
        # always passes one) but deliberately not forwarded: TransformerLM.forward() has no
        # such parameter, relying purely on causal ordering. This is only correct because
        # DPO's batches are right-padded (confirmed in Task 4) -- causal masking alone then
        # prevents any real token from attending into the padded tail, the same property
        # make_collate_fn's SFT collation already relies on.
        logits = self.model(input_ids)
        return CausalLMOutputWithPast(logits=logits)


def wrap_tokenizer(tokenizer: Tokenizer) -> PreTrainedTokenizerFast:
    # eos_token="[PAD]" matters beyond labeling: TRL's DPOTrainer appends
    # tokenizer.eos_token (literal text) to chosen/rejected completions before encoding,
    # which is exactly the stop-signal role [PAD] already plays in this repo's SFT
    # convention (see tests/test_tokenizer.py's PAD-round-trip regression test).
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        eos_token=PAD_TOKEN,
    )
