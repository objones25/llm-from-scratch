import pytest
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


def test_preallocated_cache_matches_dynamic_cache_across_multiple_updates():
    # max_seq_len is an opt-in perf path (preallocate + write in-place instead of
    # torch.cat-growing every step); must produce bit-identical results to the default
    # dynamic-growth path.
    torch.manual_seed(0)
    dynamic_cache = KVCache()
    prealloc_cache = KVCache(max_seq_len=10)
    for _ in range(4):
        k = torch.randn(1, 2, 1, 4)
        v = torch.randn(1, 2, 1, 4)
        k_dyn, v_dyn = dynamic_cache.update(layer_idx=0, k=k.clone(), v=v.clone())
        k_pre, v_pre = prealloc_cache.update(layer_idx=0, k=k.clone(), v=v.clone())
        assert torch.equal(k_dyn, k_pre)
        assert torch.equal(v_dyn, v_pre)
    assert dynamic_cache.seq_len == prealloc_cache.seq_len == 4


def test_preallocated_cache_supports_multi_token_prefill_then_single_token_steps():
    cache = KVCache(max_seq_len=8)
    k_prefill = torch.randn(1, 2, 5, 4)
    v_prefill = torch.randn(1, 2, 5, 4)
    k_out, v_out = cache.update(layer_idx=0, k=k_prefill, v=v_prefill)
    assert torch.equal(k_out, k_prefill)
    assert torch.equal(v_out, v_prefill)
    assert cache.seq_len == 5

    k_step = torch.randn(1, 2, 1, 4)
    v_step = torch.randn(1, 2, 1, 4)
    k_out, v_out = cache.update(layer_idx=0, k=k_step, v=v_step)
    assert k_out.shape == (1, 2, 6, 4)
    assert torch.equal(k_out[:, :, :5, :], k_prefill)
    assert torch.equal(k_out[:, :, 5:, :], k_step)
    assert cache.seq_len == 6


def test_preallocated_cache_layers_are_independent():
    cache = KVCache(max_seq_len=8)
    cache.update(layer_idx=0, k=torch.randn(1, 2, 2, 4), v=torch.randn(1, 2, 2, 4))
    cache.update(layer_idx=1, k=torch.randn(1, 2, 2, 4), v=torch.randn(1, 2, 2, 4))
    k0_out, _ = cache.update(layer_idx=0, k=torch.randn(1, 2, 1, 4), v=torch.randn(1, 2, 1, 4))
    assert k0_out.shape == (1, 2, 3, 4)


def test_preallocated_cache_raises_on_overflow():
    cache = KVCache(max_seq_len=2)
    cache.update(layer_idx=0, k=torch.randn(1, 2, 1, 4), v=torch.randn(1, 2, 1, 4))
    cache.update(layer_idx=0, k=torch.randn(1, 2, 1, 4), v=torch.randn(1, 2, 1, 4))
    with pytest.raises(ValueError):
        cache.update(layer_idx=0, k=torch.randn(1, 2, 1, 4), v=torch.randn(1, 2, 1, 4))


def test_preallocated_cache_seq_len_is_zero_when_empty():
    cache = KVCache(max_seq_len=5)
    assert cache.seq_len == 0
