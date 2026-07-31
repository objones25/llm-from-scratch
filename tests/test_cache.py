import torch

from llmtrain.model.cache import KVCache


def test_update_returns_new_kv_when_cache_empty():
    cache = KVCache()
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    k_out, v_out = cache.update(layer_idx=0, k=k, v=v)
    assert torch.equal(k_out, k)
    assert torch.equal(v_out, v)
    assert cache.seq_len == 3


def test_update_concatenates_across_calls():
    cache = KVCache()
    k1 = torch.randn(1, 2, 3, 4)
    v1 = torch.randn(1, 2, 3, 4)
    cache.update(layer_idx=0, k=k1, v=v1)
    k2 = torch.randn(1, 2, 1, 4)
    v2 = torch.randn(1, 2, 1, 4)
    k_out, _v_out = cache.update(layer_idx=0, k=k2, v=v2)
    assert k_out.shape == (1, 2, 4, 4)
    assert torch.equal(k_out[:, :, :3, :], k1)
    assert torch.equal(k_out[:, :, 3:, :], k2)
    assert cache.seq_len == 4


def test_layers_are_cached_independently():
    cache = KVCache()
    cache.update(layer_idx=0, k=torch.randn(1, 2, 2, 4), v=torch.randn(1, 2, 2, 4))
    cache.update(layer_idx=1, k=torch.randn(1, 2, 2, 4), v=torch.randn(1, 2, 2, 4))
    k0_out, _ = cache.update(layer_idx=0, k=torch.randn(1, 2, 1, 4), v=torch.randn(1, 2, 1, 4))
    assert k0_out.shape == (1, 2, 3, 4)


def test_seq_len_is_zero_for_empty_cache():
    cache = KVCache()
    assert cache.seq_len == 0
