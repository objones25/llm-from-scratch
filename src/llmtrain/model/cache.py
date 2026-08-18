import torch


class KVCache:
    def __init__(self, max_seq_len: int | None = None) -> None:
        # max_seq_len is opt-in: when set, buffers are preallocated once per layer and
        # written in-place, avoiding a torch.cat reallocate+copy on every decode step.
        # Left unset, behavior is unchanged from before (dynamic torch.cat growth) --
        # callers that don't know an upper bound up front (or existing tests) are
        # unaffected.
        self._max_seq_len = max_seq_len
        self._entries: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._filled_len: dict[int, int] = {}

    @property
    def seq_len(self) -> int:
        if not self._entries:
            return 0
        first_layer_idx = next(iter(self._entries))
        if self._max_seq_len is not None:
            return self._filled_len[first_layer_idx]
        first_k, _ = self._entries[first_layer_idx]
        return first_k.shape[2]

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._max_seq_len is None:
            if layer_idx in self._entries:
                prev_k, prev_v = self._entries[layer_idx]
                k = torch.cat([prev_k, k], dim=2)
                v = torch.cat([prev_v, v], dim=2)
            self._entries[layer_idx] = (k, v)
            return k, v

        new_len = k.shape[2]
        if layer_idx not in self._entries:
            batch, n_kv_heads, _, head_dim = k.shape
            buf_k = torch.zeros(
                batch, n_kv_heads, self._max_seq_len, head_dim, dtype=k.dtype, device=k.device
            )
            buf_v = torch.zeros(
                batch, n_kv_heads, self._max_seq_len, head_dim, dtype=v.dtype, device=v.device
            )
            self._entries[layer_idx] = (buf_k, buf_v)
            self._filled_len[layer_idx] = 0

        buf_k, buf_v = self._entries[layer_idx]
        filled = self._filled_len[layer_idx]
        if filled + new_len > self._max_seq_len:
            raise ValueError(
                f"KVCache overflow: max_seq_len={self._max_seq_len} exceeded "
                f"(already filled {filled}, incoming {new_len})"
            )
        buf_k[:, :, filled : filled + new_len, :] = k
        buf_v[:, :, filled : filled + new_len, :] = v
        self._filled_len[layer_idx] = filled + new_len
        return buf_k[:, :, : filled + new_len, :], buf_v[:, :, : filled + new_len, :]
