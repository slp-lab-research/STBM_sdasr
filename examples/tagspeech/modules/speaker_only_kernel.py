from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class SpeakerOnlyOrderedKernel(nn.Module):
    """Apply ordered sinusoidal speaker kernels without semantic alignment.

    The module consumes only speaker hidden representations.  It estimates an
    ordered-speaker probability at every speaker timestep and uses those
    probabilities to mix fixed sinusoidal kernels in the speaker hidden space.
    """

    def __init__(
        self,
        *,
        speaker_dim: int,
        num_speakers: int,
        probability_hidden_dim: Optional[int] = None,
        probability_dropout: float = 0.1,
        kernel_scale: float = 0.1,
        learnable_scale: bool = True,
        kernel_index_start: int = 1,
    ) -> None:
        super().__init__()
        if speaker_dim <= 0 or num_speakers <= 0:
            raise ValueError("speaker_dim and num_speakers must be positive")
        hidden_dim = probability_hidden_dim or speaker_dim
        if hidden_dim <= 0:
            raise ValueError("probability_hidden_dim must be positive")
        if not 0.0 <= probability_dropout < 1.0:
            raise ValueError("probability_dropout must be in [0, 1)")

        self.speaker_dim = int(speaker_dim)
        self.num_speakers = int(num_speakers)
        self.speaker_probability_head = nn.Sequential(
            nn.LayerNorm(speaker_dim),
            nn.Linear(speaker_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(probability_dropout),
            nn.Linear(hidden_dim, num_speakers),
        )
        self.register_buffer(
            "speaker_kernels",
            self._build_sinusoidal_kernels(
                num_speakers, speaker_dim, kernel_index_start
            ),
            persistent=True,
        )
        scale = torch.tensor(float(kernel_scale))
        if learnable_scale:
            self.kernel_scale = nn.Parameter(scale)
        else:
            self.register_buffer("kernel_scale", scale, persistent=True)

    @staticmethod
    def _build_sinusoidal_kernels(
        num_speakers: int, speaker_dim: int, kernel_index_start: int
    ) -> torch.Tensor:
        speaker_indices = torch.arange(
            kernel_index_start,
            kernel_index_start + num_speakers,
            dtype=torch.float32,
        ).unsqueeze(1)
        feature_indices = torch.arange(
            1, speaker_dim + 1, dtype=torch.float32
        ).unsqueeze(0)
        return torch.sin(
            2.0 * math.pi * speaker_indices * feature_indices / speaker_dim
        )

    def forward(
        self,
        speaker_features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if speaker_features.ndim != 3:
            raise ValueError("speaker_features must have shape [B, T_spk, D_spk]")
        if speaker_features.size(-1) != self.speaker_dim:
            raise ValueError(
                f"Expected speaker dimension {self.speaker_dim}, "
                f"received {speaker_features.size(-1)}"
            )
        if padding_mask is not None:
            if padding_mask.dtype is not torch.bool:
                raise ValueError("padding_mask must have dtype torch.bool")
            if padding_mask.shape != speaker_features.shape[:2]:
                raise ValueError("padding_mask must have shape [B, T_spk]")

        probability_dtype = self.speaker_probability_head[1].weight.dtype
        speaker_logits = self.speaker_probability_head(
            speaker_features.to(dtype=probability_dtype)
        )
        speaker_probs = torch.sigmoid(speaker_logits)
        if padding_mask is not None:
            speaker_probs = speaker_probs.masked_fill(
                padding_mask.unsqueeze(-1), 0.0
            )

        kernel_encoding = torch.matmul(
            speaker_probs.to(dtype=speaker_features.dtype),
            self.speaker_kernels.to(
                device=speaker_features.device, dtype=speaker_features.dtype
            ),
        )
        scaled_kernel = self.kernel_scale.to(
            device=speaker_features.device, dtype=speaker_features.dtype
        ) * kernel_encoding
        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(-1).to(scaled_kernel.dtype)
            scaled_kernel = scaled_kernel * valid

        enhanced_speaker = speaker_features + scaled_kernel
        if padding_mask is not None:
            enhanced_speaker = enhanced_speaker.masked_fill(
                padding_mask.unsqueeze(-1), 0.0
            )
        return enhanced_speaker, {
            "speaker_logits": speaker_logits,
            "speaker_probs": speaker_probs,
            "kernel_encoding": kernel_encoding,
            "scaled_kernel_encoding": scaled_kernel,
            "kernel_scale": self.kernel_scale,
        }
