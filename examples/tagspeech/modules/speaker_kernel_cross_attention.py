from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .speaker_only_kernel import SpeakerOnlyOrderedKernel


class SpeakerKernelCrossAttentionConditioner(nn.Module):
    """Using residual connection to solve interjection problem, kernel for boundary reinforcement
       Using cross attention to solve the problem of speaker and semantic feature misalignment

    Pipeline:
      H_spk -> Conv1d -> GELU -> Conv1d -> ordered kernel -> boundary head
      H_sem (query), enhanced H_spk (key/value) -> cross-attention -> MLP adapter

    Semantic and speaker sequences retain their independent temporal grids
    """

    def __init__(
        self,
        *,
        semantic_dim: int,
        speaker_dim: int,
        num_speakers: int,
        attention_heads: int = 8,
        attention_dropout: float = 0.0,
        temporal_kernel_size: int = 3,
        adapter_expansion: int = 4,
        adapter_dropout: float = 0.1,
        use_temporal_convolution: bool = True,
        use_speaker_kernel: bool = True,
        use_cross_attention: bool = True,
        probability_hidden_dim: int | None = None,
        probability_dropout: float = 0.1,
        kernel_scale: float = 0.1,
        learnable_scale: bool = True,
        use_boundary_head: bool = True,
    ) -> None:
        super().__init__()
        if semantic_dim != speaker_dim:
            raise ValueError("semantic_dim and speaker_dim must match")
        if semantic_dim % attention_heads != 0:
            raise ValueError("semantic_dim must be divisible by attention_heads")
        if temporal_kernel_size <= 0 or temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd integer")
        if adapter_expansion <= 0:
            raise ValueError("adapter_expansion must be positive")

        self.use_temporal_convolution = bool(use_temporal_convolution)
        self.use_speaker_kernel = bool(use_speaker_kernel)
        self.use_cross_attention = bool(use_cross_attention)

        # === H_SPK CONVOLUTIONAL RESIDUAL INITIALIZATION START ===
        # Residual temporal block on H_spk. The ordered kernel has exactly one
        # input: the output of this speaker-side residual connection.
        hidden_dim = semantic_dim
        padding = temporal_kernel_size // 2
        self.temporal_convolution = (
            nn.Sequential(
                nn.Conv1d(
                    hidden_dim, hidden_dim, temporal_kernel_size, padding=padding
                ),
                nn.GELU(),
                nn.Conv1d(
                    hidden_dim, hidden_dim, temporal_kernel_size, padding=padding
                ),
            )
            if self.use_temporal_convolution
            else None
        )
        # === H_SPK CONVOLUTIONAL RESIDUAL INITIALIZATION END ===

        #kernel
        self.kernel = (
            SpeakerOnlyOrderedKernel(
                speaker_dim=speaker_dim,
                num_speakers=num_speakers,
                probability_hidden_dim=probability_hidden_dim,
                probability_dropout=probability_dropout,
                kernel_scale=kernel_scale,
                learnable_scale=learnable_scale,
            )
            if self.use_speaker_kernel
            else None
        )

        #fusion of semantic and speaker features
        self.cross_attention = (
            nn.MultiheadAttention(
                hidden_dim,
                attention_heads,
                dropout=attention_dropout,
                batch_first=True,
            )
            if self.use_cross_attention
            else None
        )

        #MLP
        adapter_hidden = hidden_dim * adapter_expansion
        self.adapter = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, adapter_hidden),
                nn.GELU(),
                nn.Dropout(adapter_dropout),
                nn.Linear(adapter_hidden, hidden_dim),
                nn.Dropout(adapter_dropout),
            )
            if self.use_cross_attention
            else None
        )

        #boundary head for speaker features
        self.boundary_head = (
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
            if use_boundary_head
            else None
        )

    def forward(
        self,
        semantic_features: torch.Tensor,
        semantic_lens: torch.Tensor,
        speaker_features: torch.Tensor,
        speaker_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_inputs(
            semantic_features, semantic_lens, speaker_features, speaker_lens
        )
        semantic_mask = self._padding_mask(
            semantic_lens, semantic_features.size(1)
        )
        speaker_mask = self._padding_mask(speaker_lens, speaker_features.size(1))
        if (speaker_lens <= 0).any():
            raise ValueError(
                "Cross-attention requires at least one valid speaker frame per sample; "
                f"speaker_lens={speaker_lens.detach().cpu().tolist()}"
            )

        speaker_input = speaker_features.masked_fill(
            speaker_mask.unsqueeze(-1), 0.0
        )
        # === H_SPK CONVOLUTIONAL RESIDUAL FORWARD START ===
        # Speaker-side residual temporal block.
        if self.use_temporal_convolution:
            temporal_dtype = self.temporal_convolution[0].weight.dtype
            temporal_delta = self.temporal_convolution(
                speaker_input.to(dtype=temporal_dtype).transpose(1, 2)
            ).transpose(1, 2)
            temporal_speaker = speaker_input + temporal_delta.to(
                dtype=speaker_features.dtype
            )
        else:
            # Kernel-only ablation: bypass Conv1D -> GELU -> Conv1D and its
            # residual addition. The ordered kernel consumes H_spk directly.
            temporal_speaker = speaker_input
        temporal_speaker = temporal_speaker.masked_fill(
            speaker_mask.unsqueeze(-1), 0.0
        )
        # === H_SPK CONVOLUTIONAL RESIDUAL FORWARD END ===

        # The kernel consumes either H_spk directly or its temporal enhancement.
        if self.use_speaker_kernel:
            enhanced_speaker, outputs = self.kernel(
                temporal_speaker, padding_mask=speaker_mask
            )
        else:
            # Boundary-only ablation: the ordered speaker kernel is bypassed.
            enhanced_speaker, outputs = temporal_speaker, {}

        if self.use_cross_attention:
            attention_dtype = self.cross_attention.in_proj_weight.dtype
            attended, _ = self.cross_attention(
                query=semantic_features.to(dtype=attention_dtype),
                key=enhanced_speaker.to(dtype=attention_dtype),
                value=enhanced_speaker.to(dtype=attention_dtype),
                key_padding_mask=speaker_mask,
                need_weights=False,
            )
            attended = attended.to(dtype=semantic_features.dtype)
            adapter_dtype = self.adapter[1].weight.dtype
            adapter_delta = self.adapter(attended.to(dtype=adapter_dtype)).to(
                dtype=semantic_features.dtype
            )
            conditioned_semantic = semantic_features + adapter_delta
        else:
            # Boundary-only ablation: do not condition the semantic stream.
            conditioned_semantic = semantic_features
        conditioned_semantic = conditioned_semantic.masked_fill(
            semantic_mask.unsqueeze(-1), 0.0
        )

        outputs = dict(outputs)
        outputs["enhanced_speaker_features"] = enhanced_speaker
        outputs["speaker_padding_mask"] = speaker_mask
        outputs["segment_lens"] = speaker_lens
        outputs["segment_padding_mask"] = speaker_mask
        if self.boundary_head is not None:
            boundary_dtype = self.boundary_head[1].weight.dtype
            boundary_logits = self.boundary_head(
                enhanced_speaker.to(dtype=boundary_dtype)
            ).squeeze(-1)
            boundary_logits = boundary_logits.masked_fill(speaker_mask, 0.0)
            outputs["boundary_logits"] = boundary_logits
            outputs["boundary_probs"] = torch.sigmoid(boundary_logits).masked_fill(
                speaker_mask, 0.0
            )
        return conditioned_semantic, semantic_lens, outputs

    @staticmethod
    def _padding_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
        positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
        return positions >= lengths.unsqueeze(1)

    @staticmethod
    def _validate_inputs(
        semantic_features: torch.Tensor,
        semantic_lens: torch.Tensor,
        speaker_features: torch.Tensor,
        speaker_lens: torch.Tensor,
    ) -> None:
        if semantic_features.ndim != 3 or speaker_features.ndim != 3:
            raise ValueError("feature streams must have shape [B, T, D]")
        if semantic_features.size(0) != speaker_features.size(0):
            raise ValueError("semantic and speaker batch sizes must match")
        batch_shape = (semantic_features.size(0),)
        if semantic_lens.shape != batch_shape or speaker_lens.shape != batch_shape:
            raise ValueError("feature lengths must have shape [B]")
