import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedSpeakerKernel(nn.Module):
    """Inject ordered speaker information into post-anchor semantic features.

    Expected speaker indexing:
        speaker 0 = first speaker to appear
        speaker 1 = second speaker to appear

    Inputs:
        semantic_anchors: [B, N, D_sem]
        speaker_anchors:  [B, N, D_spk]

    Output:
        conditioned_semantic: [B, N, D_sem]
    """

    def __init__(
            self,
            semantic_dim: int,
            num_speakers: int,
            speaker_dim: Optional[int] = None,
            probability_hidden_dim: Optional[int] = None,
            probability_dropout: float = 0.1,
            kernel_scale: float = 0.1,
            learnable_scale: bool = True,
            normalize_semantic: bool = True,
            kernel_index_start: int = 1,
    ) -> None:
        super().__init__()

        if semantic_dim <= 0:
            raise ValueError("semantic dim must be positive")

        if num_speakers <= 0:
            raise ValueError("the number of speakers must be larger than 0")

        speaker_dim = speaker_dim or semantic_dim
        probability_hidden_dim = probability_hidden_dim or speaker_dim

        if speaker_dim <= 0:
            raise ValueError("speaker_dim must be positive")

        if probability_hidden_dim <= 0:
            raise ValueError("probability_hidden_dim must be positive")

        if not 0.0 <= probability_dropout < 1.0:
            raise ValueError("probability_dropout must be in [0, 1)")

        self.semantic_dim = semantic_dim
        self.speaker_dim = speaker_dim
        self.num_speakers = num_speakers
        self.normalize_semantic = normalize_semantic

        # The output channels have a fixed meaning: channel 0 is the first speaker, channel 1 the second speaker, and so on.
        self.speaker_probability_head = nn.Sequential(
            nn.LayerNorm(speaker_dim),
            nn.Linear(speaker_dim, probability_hidden_dim),
            nn.GELU(),
            nn.Dropout(probability_dropout),
            nn.Linear(probability_hidden_dim, num_speakers),
        )

        kernels = self._build_sinusoidal_kernels(
            num_speakers=self.num_speakers,
            semantic_dim=semantic_dim,
            kernel_index_start=kernel_index_start,
        )

        self.register_buffer(
            "speaker_kernels",
            kernels,
            persistent=True,
        )

        scale = torch.tensor(float(kernel_scale))

        if learnable_scale:
            self.kernel_scale = nn.Parameter(scale)
        else:
            self.register_buffer(
                "kernel_scale",
                scale,
                persistent=True,
            )

    @staticmethod
    def _build_sinusoidal_kernels(
        num_speakers: int,
        semantic_dim: int,
        kernel_index_start: int,
    ) -> torch.Tensor:
        '''
        kappa_{k,z} = sin(2*pi*k*z / M)

        where:
            k = speaker index
            z = embedding dimension index
            M = semantic embedding dimension
        '''

        #k values
        speaker_indices = torch.arange(
            kernel_index_start,
            kernel_index_start + num_speakers,
            dtype=torch.float32,
        ).unsqueeze(1)

        #z values
        feature_indices = torch.arange(
            1,
            semantic_dim + 1,
            dtype=torch.float32,
        ).unsqueeze(0)

        kernels = torch.sin(
                2.0
                * math.pi
                * speaker_indices
                * feature_indices
                / semantic_dim
        )

        return kernels


    def forward(
        self,
        semantic_anchors: torch.Tensor,
        speaker_anchors: Optional[torch.Tensor] = None,
        speaker_logits: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if speaker_logits is None:
            if speaker_anchors is None:
                raise ValueError(
                    "FixedSpeakerKernel requires post-anchor speaker features "
                    "with shape [B, N, D_spk]"
                )

            self._validate_anchor_shapes(
                semantic_anchors=semantic_anchors,
                speaker_anchors=speaker_anchors,
            )

            probability_dtype = self.speaker_probability_head[1].weight.dtype
            speaker_logits = self.speaker_probability_head(
                speaker_anchors.to(dtype=probability_dtype)
            )

        self._validate_logit_shapes(
            semantic_anchors=semantic_anchors,
            speaker_logits=speaker_logits,
        )

        speaker_probs = torch.sigmoid(speaker_logits)

        if padding_mask is not None:
            self._validate_padding_mask(semantic_anchors, padding_mask)
            speaker_probs = speaker_probs.masked_fill(
                padding_mask.unsqueeze(-1), 0.0
            )

        kernel_encoding = torch.matmul(
            speaker_probs.to(dtype=semantic_anchors.dtype),
            self.speaker_kernels.to(
                dtype=semantic_anchors.dtype,
                device=semantic_anchors.device,
            )
        )

        if self.normalize_semantic:
            semantic_base = F.normalize(
                semantic_anchors,
                p=2,
                dim=-1,
            )

        else:
            semantic_base = semantic_anchors

        scaled_kernel = self.kernel_scale.to(
            dtype=semantic_anchors.dtype,
            device=semantic_anchors.device,
        ) * kernel_encoding

        if padding_mask is not None:
            valid_mask = (~padding_mask).unsqueeze(-1)
            scaled_kernel = scaled_kernel * valid_mask.to(
                scaled_kernel.dtype
            )

        conditioned_semantic = semantic_base + scaled_kernel

        outputs = {
            "speaker_probs": speaker_probs,
            "kernel_encoding": kernel_encoding,
            "scaled_kernel_encoding": scaled_kernel,
            "kernel_scale": self.kernel_scale,
        }

        return conditioned_semantic, outputs


    def _validate_anchor_shapes(
            self,
            semantic_anchors: torch.Tensor,
            speaker_anchors: torch.Tensor,
    ) -> None:

        if semantic_anchors.ndim != 3:
            raise ValueError(
                "semantic_anchors must have shape [B, N, D_sem]"
            )

        if speaker_anchors.ndim != 3:
            raise ValueError(
                "speaker_anchors must have shape [B, N, D_spk]"
            )

        if semantic_anchors.shape[:2] != speaker_anchors.shape[:2]:
            raise ValueError(
                "semantic_anchors and speaker_anchors must have the same "
                "batch and anchor dimensions"
            )

        if semantic_anchors.shape[-1] != self.semantic_dim:
            raise ValueError(
                f"Expected semantic dimension {self.semantic_dim}, "
                f"received {semantic_anchors.shape[-1]}"
            )

        if speaker_anchors.shape[-1] != self.speaker_dim:
            raise ValueError(
                f"Expected speaker dimension {self.speaker_dim}, "
                f"received {speaker_anchors.shape[-1]}"
            )

    def _validate_logit_shapes(
            self,
            semantic_anchors: torch.Tensor,
            speaker_logits: torch.Tensor,
    ) -> None:

        if semantic_anchors.ndim != 3:
            raise ValueError(
                "semantic_anchors must have shape [B, N, D_sem]"
            )

        if speaker_logits.ndim != 3:
            raise ValueError(
                "speaker_logits must have shape [B, N, K]"
            )

        if semantic_anchors.shape[:2] != speaker_logits.shape[:2]:
            raise ValueError(
                "semantic_anchors and speaker_logits must have"
                "the same batch and anchor dimensions"
            )

        if semantic_anchors.shape[-1] != self.semantic_dim:
            raise ValueError(
                f"Expected semantic dimension {self.semantic_dim},"
                f"received {semantic_anchors.shape[-1]}"
            )

        if speaker_logits.shape[-1] != self.num_speakers:
            raise ValueError(
                f"Expected {self.num_speakers} speaker channels, "
                f"received {speaker_logits.shape[-1]}"
            )

    @staticmethod
    def _validate_padding_mask(
        semantic_anchors: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> None:

        if padding_mask.dtype is not torch.bool:
            raise ValueError("padding_mask must have dtype torch.bool")

        if padding_mask.shape != semantic_anchors.shape[:2]:
            raise ValueError(
                "padding_mask must have shape [B, N] matching the anchors"
            )
