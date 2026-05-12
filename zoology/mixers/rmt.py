from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class _RMTBlock(nn.Module):
    """Small Transformer block used inside an RMT segment."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        mlp_hidden_mult: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        hidden_dim = d_model * mlp_hidden_mult
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout1(h)

        h = self.norm2(x)
        return x + self.dropout2(self.mlp(h))


class BaseRMTMixer(nn.Module):
    """
    Recurrent Memory Transformer mixer.

    The mixer splits a long sequence into segments. By default each segment is
    processed as `[read memory, segment tokens, write memory]`; write memory
    outputs are carried to the next segment, while only segment token outputs are
    returned. The older prefix-only `[memory, segment]` layout is available with
    `memory_layout="prefix"`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 1,
        n_layers: int = 1,
        n_mem: int = 4,
        segment_len: int = 64,
        dropout: float = 0.0,
        detach_memory_between_segments: bool = True,
        memory_layout: str = "read_write",
        mlp_hidden_mult: int = 4,
        bias: bool = True,
        layer_idx: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        if segment_len <= 0:
            raise ValueError("segment_len must be positive")
        if n_mem <= 0:
            raise ValueError("n_mem must be positive")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_mem = n_mem
        self.segment_len = segment_len
        self.detach_memory_between_segments = detach_memory_between_segments
        if memory_layout not in {"read_write", "prefix"}:
            raise ValueError("memory_layout must be 'read_write' or 'prefix'")
        self.memory_layout = memory_layout
        self.layer_idx = layer_idx

        self.memory_tokens = nn.Parameter(torch.empty(n_mem, d_model))
        nn.init.normal_(self.memory_tokens, std=0.02)

        self.blocks = nn.ModuleList(
            [
                _RMTBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    mlp_hidden_mult=mlp_hidden_mult,
                    bias=bias,
                )
                for _ in range(n_layers)
            ]
        )

        self.last_memory_norm = None
        self.last_memory_update_norm = None
        self.last_memory_grad_norm = None

    def _initial_memory(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        memory_tokens = self.memory_tokens.to(device=x.device, dtype=x.dtype)
        return memory_tokens.unsqueeze(0).expand(batch_size, -1, -1)

    def _make_attention_mask(self, segment_len: int, device: torch.device) -> torch.Tensor:
        if self.memory_layout == "prefix":
            return self._make_prefix_attention_mask(segment_len, device)
        return self._make_read_write_attention_mask(segment_len, device)

    def _make_prefix_attention_mask(
        self,
        segment_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_len = self.n_mem + segment_len
        mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)

        # Segment tokens attend to memory and causal in-segment tokens only.
        future_token_mask = torch.triu(
            torch.ones(segment_len, segment_len, dtype=torch.bool, device=device),
            diagonal=1,
        )
        mask[self.n_mem :, self.n_mem :] = future_token_mask
        return mask

    def _make_read_write_attention_mask(
        self,
        segment_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_len = self.n_mem + segment_len + self.n_mem
        mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)

        read_start = 0
        segment_start = self.n_mem
        write_start = self.n_mem + segment_len

        # Read memory is previous-segment state. It must not read current segment
        # tokens or write memory, otherwise future segment information can leak
        # back into token outputs through multi-layer internal blocks.
        mask[read_start:segment_start, segment_start:] = True

        # Segment tokens read previous memory and causal in-segment tokens, but
        # never write memory. Write memory can summarize the full segment later.
        future_token_mask = torch.triu(
            torch.ones(segment_len, segment_len, dtype=torch.bool, device=device),
            diagonal=1,
        )
        mask[segment_start:write_start, segment_start:write_start] = future_token_mask
        mask[segment_start:write_start, write_start:] = True

        # Write memory rows remain unmasked. They can attend to read memory, all
        # segment tokens, and write memory tokens because their outputs are only
        # used as memory for the next segment.
        return mask

    def _run_segment(
        self,
        memory: torch.Tensor,
        segment: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.memory_layout == "prefix":
            hidden_states = torch.cat([memory, segment], dim=1)
        else:
            hidden_states = torch.cat([memory, segment, memory], dim=1)

        attn_mask = self._make_attention_mask(segment.shape[1], segment.device)
        for block in self.blocks:
            hidden_states = block(hidden_states, attn_mask=attn_mask)

        if self.memory_layout == "prefix":
            return hidden_states[:, : self.n_mem], hidden_states[:, self.n_mem :]

        segment_start = self.n_mem
        write_start = self.n_mem + segment.shape[1]
        next_memory = hidden_states[:, write_start:]
        segment_output = hidden_states[:, segment_start:write_start]
        return next_memory, segment_output

    def _record_memory_norm(self, memory: torch.Tensor):
        with torch.no_grad():
            self.last_memory_norm = memory.detach().norm(dim=-1).mean().item()

    def get_diagnostics(self) -> Dict[str, float]:
        metrics = {}
        if self.last_memory_norm is not None:
            metrics["rmt_memory_norm"] = self.last_memory_norm
        if self.last_memory_update_norm is not None:
            metrics["rmt_memory_update_norm"] = self.last_memory_update_norm
        if self.last_memory_grad_norm is not None:
            metrics["rmt_memory_grad_norm"] = self.last_memory_grad_norm
        return metrics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 0:
            return x

        memory = self._initial_memory(x)
        outputs = []

        for start in range(0, x.shape[1], self.segment_len):
            end = min(start + self.segment_len, x.shape[1])
            segment = x[:, start:end]
            memory, segment_output = self._run_segment(memory, segment)
            outputs.append(segment_output)

            if self.detach_memory_between_segments and end < x.shape[1]:
                memory = memory.detach()

        self._record_memory_norm(memory)
        return torch.cat(outputs, dim=1)

    def state_size(self, batch_size: int = 1, sequence_length: int = 2048):
        return self.n_mem * self.d_model


class FastMemoryRMTMixer(BaseRMTMixer):
    """
    RMT variant with an explicit fast memory tensor.

    The trainer calls `update_fast_memory()` after loss.backward(). Slow model
    parameters are stepped separately and can be throttled with
    `rmt_slow_update_freq`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 1,
        n_layers: int = 1,
        n_mem: int = 4,
        segment_len: int = 64,
        dropout: float = 0.0,
        detach_memory_between_segments: bool = True,
        memory_layout: str = "read_write",
        mlp_hidden_mult: int = 4,
        bias: bool = True,
        rmt_fast_lr: float = 0.1,
        rmt_slow_update_freq: int = 1,
        rmt_clip_memory_grad: Optional[float] = 1.0,
        reset_memory_each_batch: bool = True,
        reset_memory_each_epoch: bool = False,
        layer_idx: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_mem=n_mem,
            segment_len=segment_len,
            dropout=dropout,
            detach_memory_between_segments=detach_memory_between_segments,
            memory_layout=memory_layout,
            mlp_hidden_mult=mlp_hidden_mult,
            bias=bias,
            layer_idx=layer_idx,
            **kwargs,
        )
        if rmt_slow_update_freq <= 0:
            raise ValueError("rmt_slow_update_freq must be positive")

        memory = self.memory_tokens.detach().clone()
        del self._parameters["memory_tokens"]
        self.initial_memory_tokens = nn.Parameter(memory)
        self.fast_memory_tokens = nn.Parameter(memory.clone())

        self.rmt_fast_lr = rmt_fast_lr
        self.rmt_slow_update_freq = rmt_slow_update_freq
        self.rmt_clip_memory_grad = rmt_clip_memory_grad
        self.reset_memory_each_batch = reset_memory_each_batch
        self.reset_memory_each_epoch = reset_memory_each_epoch

    def fast_memory_parameters(self):
        return [self.fast_memory_tokens]

    def _reset_fast_memory(self):
        with torch.no_grad():
            self.fast_memory_tokens.copy_(self.initial_memory_tokens.detach())
        self.fast_memory_tokens.grad = None

    def on_epoch_start(self, epoch_idx: int):
        if self.training and self.reset_memory_each_epoch:
            self._reset_fast_memory()

    def on_batch_start(self, epoch_idx: int, batch_idx: int):
        if self.training and self.reset_memory_each_batch:
            self._reset_fast_memory()

    def _initial_memory(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        if self.training:
            memory_tokens = (
                self.fast_memory_tokens
                + self.initial_memory_tokens
                - self.initial_memory_tokens.detach()
            )
        else:
            memory_tokens = self.initial_memory_tokens

        memory_tokens = memory_tokens.to(device=x.device, dtype=x.dtype)
        return memory_tokens.unsqueeze(0).expand(batch_size, -1, -1)

    def update_fast_memory(self):
        grad = self.fast_memory_tokens.grad
        if grad is None:
            self.last_memory_grad_norm = 0.0
            self.last_memory_update_norm = 0.0
            return

        with torch.no_grad():
            grad_for_update = grad
            grad_norm = grad_for_update.norm()
            self.last_memory_grad_norm = grad_norm.item()

            if self.rmt_clip_memory_grad is not None and self.rmt_clip_memory_grad > 0:
                clip = torch.as_tensor(
                    self.rmt_clip_memory_grad,
                    dtype=grad_for_update.dtype,
                    device=grad_for_update.device,
                )
                scale = torch.clamp(clip / (grad_norm + 1e-12), max=1.0)
                grad_for_update = grad_for_update * scale

            update = -self.rmt_fast_lr * grad_for_update
            self.fast_memory_tokens.add_(update)
            self.last_memory_update_norm = update.norm().item()
            self.last_memory_norm = self.fast_memory_tokens.norm(dim=-1).mean().item()

        self.fast_memory_tokens.grad = None
