from __future__ import annotations

from transformers import AutoConfig as HFConfig
from transformers import PretrainedConfig

from ...auto.auto_config import AutoConfig
from ..base.model_config import BaseConfig


class LalmConfig(BaseConfig):
    """Configuration for Large Audio-Language Model (LALM)."""

    model_type: str = "lalm"

    def __init__(
        self,
        *,
        audio_encoder_config: dict | BaseConfig | PretrainedConfig | None = None,
        llm_config: dict | PretrainedConfig | None = None,
        use_flash_attn: bool = False,
        audio_encoder_projector_ds_rate: int = 8,
        exclude_from_checkpoint: list | None = None,
        tag_audio_boundary: bool = False,
        audio_token: str = "<|AUDIO|>",
        **kwargs,
    ):
        # Audio encoder config
        if isinstance(audio_encoder_config, dict):
            try:
                audio_encoder_config = AutoConfig.for_model(**audio_encoder_config)
            except ValueError:
                audio_encoder_config = HFConfig.for_model(**audio_encoder_config)
        elif audio_encoder_config is None:
            from ...models.zipformer.model_config import ZipformerConfig

            audio_encoder_config = ZipformerConfig(
                encoder_dim=[192, 256, 512, 768, 512, 256],
                feedforward_dim=[576, 768, 1536, 2304, 1536, 768],
                num_encoder_layers=[2, 2, 4, 5, 4, 2],
            )
        self.audio_encoder_config = audio_encoder_config

        # LLM config (HF)
        if isinstance(llm_config, dict):
            if llm_config.get("model_type") == "qwen2_5_omni_thinker":
                from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
                    Qwen2_5OmniThinkerConfig,
                )

                llm_config = Qwen2_5OmniThinkerConfig.from_dict(llm_config)
            else:
                llm_config = HFConfig.for_model(**llm_config)
        elif llm_config is None:
            from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

            llm_config = Qwen2Config()
        self.llm_config = llm_config

        # Misc settings
        self.use_flash_attn = use_flash_attn
        self.audio_encoder_projector_ds_rate = audio_encoder_projector_ds_rate
        self.exclude_from_checkpoint = exclude_from_checkpoint
        self.tag_audio_boundary = tag_audio_boundary
        self.audio_token = audio_token

        super().__init__(**kwargs)
