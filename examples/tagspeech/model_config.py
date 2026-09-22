from __future__ import annotations

import os

from transformers import AutoConfig as HFConfig
from transformers import PretrainedConfig

from auden.auto.auto_config import AutoConfig
from auden.models.base.model_config import BaseConfig
from auden.models.lalm.model_config import LalmConfig


class TagSpeechBaseConfig(LalmConfig):
    """Configuration for AudioLLM with dual audio tokens and separate projectors.

    Architecture:
        Audio Input
            ↓
        ┌─────────────┬─────────────┐
        │             │             │
    Semantic Encoder  Voice Encoder
        │             │
        ↓             ↓
    Projector1    Projector2
        │             │
        ↓             ↓
    [B,L1,D_llm]  [B,L2,D_llm]
        │             │
        └─────┬───────┘
              │
        Text: "text <|AUDIO|> speaker <|AUDIO|>"
              │             │
              ↓             ↓
        Semantic Embedding Voice Embedding
              │             │
              └─────┬───────┘
                    │
                  LLM
    """

    model_type: str = "tagspeech-base"

    def __init__(
        self,
        *,
        voice_encoder_config: dict = None,
        semantic_projector_ds_rate: int = 4,
        voice_projector_ds_rate: int = 4,
        qwen2_5_omni_pretrained_model: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Voice encoder config
        if voice_encoder_config is None:
            voice_encoder_config = self.audio_encoder_config
        self.voice_encoder_config = voice_encoder_config

        # Dual projector downsampling rates
        self.semantic_projector_ds_rate = semantic_projector_ds_rate
        self.voice_projector_ds_rate = voice_projector_ds_rate
        # Path to the shared frozen Qwen2.5 Omni checkpoint.
        self.qwen2_5_omni_pretrained_model = qwen2_5_omni_pretrained_model


class TagSpeechConfig(TagSpeechBaseConfig):
    """Dual-audio-tokens model that inserts numeric anchors derived from digit embeddings.

    Similar to the anchor_text model, but anchors consist of character embeddings from natural number
    sequences (1, 2, 3, ...), ensuring that semantic/voice branches insert the same numbered anchors
    at the same real-time positions for precise time alignment.

    Ordered = speaker slots assigned by first appearance in each recording
    (first speaker = 0, second speaker = 1, etc.); returning speakers keep
    their slots. Each slot has a corresponding sinusoidal kernel.
    The ordered_kernel_* names are retained for config/checkpoint compatibility.

    """

    model_type: str = "tagspeech"

    def __init__(
        self,
        *,
        semantic_anchor_interval: int = 8,
        voice_anchor_interval: int = 8,
        insert_anchors_at_ends: bool = True,
        # === SINUSOIDAL KERNEL CONFIG ===
        use_segment_speaker_kernel: bool = False,
        speaker_kernel_enabled: bool = True,
        speaker_kernel_cross_attention_enabled: bool = True,
        speaker_kernel_use_temporal_convolution: bool = True,
        ordered_kernel_num_speakers: int = 4,
        ordered_kernel_probability_hidden_dim: int | None = None,
        ordered_kernel_probability_dropout: float = 0.1,
        ordered_kernel_scale: float = 0.1,
        ordered_kernel_learnable_scale: bool = True,
        ordered_kernel_normalize_semantic: bool = False,
        use_boundary_loss: bool = False,
        # === END SINUSOIDAL KERNEL CONFIG ===
        semantic_projector_ds_rate: int = 4,
        voice_projector_ds_rate: int = 4,
        **kwargs,
    ):
        super().__init__(
            semantic_projector_ds_rate=semantic_projector_ds_rate,
            voice_projector_ds_rate=voice_projector_ds_rate,
            **kwargs,
        )

        self.semantic_anchor_interval = int(semantic_anchor_interval)
        self.voice_anchor_interval = int(voice_anchor_interval)
        self.insert_anchors_at_ends = bool(insert_anchors_at_ends)
        # === SINUSOIDAL KERNEL CONFIG ===
        self.use_segment_speaker_kernel = bool(use_segment_speaker_kernel)
        self.speaker_kernel_enabled = bool(speaker_kernel_enabled)
        self.speaker_kernel_cross_attention_enabled = bool(
            speaker_kernel_cross_attention_enabled
        )
        self.speaker_kernel_use_temporal_convolution = bool(
            speaker_kernel_use_temporal_convolution
        )
        self.ordered_kernel_num_speakers = int(ordered_kernel_num_speakers)
        self.ordered_kernel_probability_hidden_dim = (
            ordered_kernel_probability_hidden_dim
        )
        self.ordered_kernel_probability_dropout = float(
            ordered_kernel_probability_dropout
        )
        self.ordered_kernel_scale = float(ordered_kernel_scale)
        self.ordered_kernel_learnable_scale = bool(
            ordered_kernel_learnable_scale
        )
        self.ordered_kernel_normalize_semantic = bool(
            ordered_kernel_normalize_semantic
        )
        self.use_boundary_loss = bool(use_boundary_loss)

        if self.ordered_kernel_num_speakers <= 0:
            raise ValueError("ordered_kernel_num_speakers must be positive")
        # === END SINUSOIDAL KERNEL CONFIG ===
        if self.semantic_anchor_interval <= 0 or self.voice_anchor_interval <= 0:
            raise ValueError(
                f"Anchor intervals must be positive integers, got "
                f"semantic={self.semantic_anchor_interval}, voice={self.voice_anchor_interval}"
            )

        # Require the two branches to share the same real-time spacing:
        # projector_ds_rate * anchor_interval (measured on encoder frames) should match.
        semantic_stride = (
            self.semantic_anchor_interval * self.semantic_projector_ds_rate
        )
        voice_stride = self.voice_anchor_interval * self.voice_projector_ds_rate
        if semantic_stride != voice_stride:
            raise ValueError(
                "semantic_anchor_interval * semantic_projector_ds_rate must equal "
                "voice_anchor_interval * voice_projector_ds_rate to keep anchors time-aligned.\n"
                f"Got semantic: {self.semantic_anchor_interval} * {self.semantic_projector_ds_rate} = {semantic_stride}, "
                f"voice: {self.voice_anchor_interval} * {self.voice_projector_ds_rate} = {voice_stride}"
            )
