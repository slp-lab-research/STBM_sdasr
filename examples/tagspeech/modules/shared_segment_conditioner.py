from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .sin_kernel_ordered import FixedSpeakerKernel


class SharedSegmentSpeakerConditioner(nn.Module):
    """Condition semantic frames using aligned semantic/speaker segments.

    Both projected streams remain intact at the module boundary. Internally,
    corresponding real-time segments are mean pooled to form compact aligned
    representations. The ordered speaker kernel produces one delta per shared
    segment, which is broadcast back across that segment in the semantic stream.
    """

    def __init__(
        self,
        *,
        semantic_dim: int,
        speaker_dim: int,
        semantic_interval: int,
        speaker_interval: int,
        num_speakers: int,
        probability_hidden_dim: int | None = None,
        probability_dropout: float = 0.1,
        kernel_scale: float = 0.1,
        learnable_scale: bool = True,
        normalize_semantic: bool = False,
        use_boundary_head: bool = False,
    ) -> None:
        super().__init__()

        if semantic_interval <= 0 or speaker_interval <= 0:
            raise ValueError("segment intervals must be positive")

        self.semantic_interval = int(semantic_interval)
        self.speaker_interval = int(speaker_interval)
        self.kernel = FixedSpeakerKernel(
            semantic_dim=semantic_dim,
            speaker_dim=speaker_dim,
            num_speakers=num_speakers,
            probability_hidden_dim=probability_hidden_dim,
            probability_dropout=probability_dropout,
            kernel_scale=kernel_scale,
            learnable_scale=learnable_scale,
            normalize_semantic=normalize_semantic,
        )
        # One raw logit per shared segment. Sigmoid is used only to expose
        # probabilities; training uses BCE-with-logits for numerical stability.
        self.boundary_head = (
            nn.Sequential(
                nn.LayerNorm(speaker_dim),
                nn.Linear(speaker_dim, 1),
            )
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
            semantic_features,
            semantic_lens,
            speaker_features,
            speaker_lens,
        )

        semantic_segments, speaker_segments, segment_lens = self._pool_segments(
            semantic_features,
            semantic_lens,
            speaker_features,
            speaker_lens,
        )
        max_segments = semantic_segments.size(1)
        positions = torch.arange(
            max_segments, device=semantic_features.device
        ).unsqueeze(0)
        padding_mask = positions >= segment_lens.unsqueeze(1)

        _, kernel_outputs = self.kernel(
            semantic_anchors=semantic_segments,
            speaker_anchors=speaker_segments,
            padding_mask=padding_mask,
        )
        conditioned_semantic = self._broadcast_segment_deltas(
            semantic_features,
            semantic_lens,
            segment_lens,
            kernel_outputs["scaled_kernel_encoding"],
        )
        outputs = dict(kernel_outputs)
        outputs["segment_lens"] = segment_lens
        outputs["segment_padding_mask"] = padding_mask
        if self.boundary_head is not None:
            # Validation runs outside autocast while the projected streams are
            # explicitly FP16. Match the boundary head's parameter dtype so
            # LayerNorm works consistently in both training and validation.
            boundary_dtype = self.boundary_head[1].weight.dtype
            boundary_logits = self.boundary_head(
                speaker_segments.to(dtype=boundary_dtype)
            ).squeeze(-1)
            boundary_logits = boundary_logits.masked_fill(padding_mask, 0.0)
            boundary_probs = torch.sigmoid(boundary_logits).masked_fill(
                padding_mask, 0.0
            )
            outputs["boundary_logits"] = boundary_logits
            outputs["boundary_probs"] = boundary_probs
        return conditioned_semantic, semantic_lens, outputs

    def _pool_segments(
        self,
        semantic_features: torch.Tensor,
        semantic_lens: torch.Tensor,
        speaker_features: torch.Tensor,
        speaker_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = semantic_features.size(0)
        semantic_samples: List[torch.Tensor] = []
        speaker_samples: List[torch.Tensor] = []
        segment_counts: List[int] = []

        for batch_idx in range(batch_size):
            semantic_len = int(semantic_lens[batch_idx].item())
            speaker_len = int(speaker_lens[batch_idx].item())
            semantic_count = self._ceil_div(
                semantic_len, self.semantic_interval
            )
            speaker_count = self._ceil_div(speaker_len, self.speaker_interval)
            segment_count = min(semantic_count, speaker_count)
            segment_counts.append(segment_count)

            semantic_rows: List[torch.Tensor] = []
            speaker_rows: List[torch.Tensor] = []
            for segment_idx in range(segment_count):
                semantic_start = segment_idx * self.semantic_interval
                semantic_end = min(
                    semantic_start + self.semantic_interval, semantic_len
                )
                speaker_start = segment_idx * self.speaker_interval
                speaker_end = min(
                    speaker_start + self.speaker_interval, speaker_len
                )
                semantic_rows.append(
                    semantic_features[
                        batch_idx, semantic_start:semantic_end
                    ].mean(dim=0)
                )
                speaker_rows.append(
                    speaker_features[
                        batch_idx, speaker_start:speaker_end
                    ].mean(dim=0)
                )

            semantic_samples.append(
                torch.stack(semantic_rows)
                if semantic_rows
                else semantic_features[batch_idx, :0]
            )
            speaker_samples.append(
                torch.stack(speaker_rows)
                if speaker_rows
                else speaker_features[batch_idx, :0]
            )

        max_segments = max(segment_counts, default=0)
        if max_segments == 0:
            raise ValueError("No shared temporal segments were available")

        semantic_padded = semantic_features.new_zeros(
            batch_size, max_segments, semantic_features.size(-1)
        )
        speaker_padded = speaker_features.new_zeros(
            batch_size, max_segments, speaker_features.size(-1)
        )
        for batch_idx, segment_count in enumerate(segment_counts):
            if segment_count:
                semantic_padded[batch_idx, :segment_count] = semantic_samples[
                    batch_idx
                ]
                speaker_padded[batch_idx, :segment_count] = speaker_samples[
                    batch_idx
                ]

        segment_lens = torch.tensor(
            segment_counts,
            dtype=semantic_lens.dtype,
            device=semantic_features.device,
        )
        return semantic_padded, speaker_padded, segment_lens

    def _broadcast_segment_deltas(
        self,
        semantic_features: torch.Tensor,
        semantic_lens: torch.Tensor,
        segment_lens: torch.Tensor,
        segment_deltas: torch.Tensor,
    ) -> torch.Tensor:
        conditioned = semantic_features.clone()
        for batch_idx in range(semantic_features.size(0)):
            semantic_len = int(semantic_lens[batch_idx].item())
            segment_count = int(segment_lens[batch_idx].item())
            for segment_idx in range(segment_count):
                start = segment_idx * self.semantic_interval
                end = min(start + self.semantic_interval, semantic_len)
                conditioned[batch_idx, start:end] = (
                    conditioned[batch_idx, start:end]
                    + segment_deltas[batch_idx, segment_idx]
                )
        return conditioned

    @staticmethod
    def _ceil_div(value: int, divisor: int) -> int:
        return (value + divisor - 1) // divisor

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
        expected_lens_shape = (semantic_features.size(0),)
        if semantic_lens.shape != expected_lens_shape:
            raise ValueError("semantic_lens must have shape [B]")
        if speaker_lens.shape != expected_lens_shape:
            raise ValueError("speaker_lens must have shape [B]")
