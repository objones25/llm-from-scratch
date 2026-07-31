import torch


class KVCache:
    def __init__(self) -> None:
        self._entries: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def seq_len(self) -> int:
        if not self._entries:
            return 0
        first_k, _ = next(iter(self._entries.values()))
        return first_k.shape[2]

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx in self._entries:
            prev_k, prev_v = self._entries[layer_idx]
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        self._entries[layer_idx] = (k, v)
        return k, v
