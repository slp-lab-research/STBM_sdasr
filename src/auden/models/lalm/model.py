from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer as HFTokenizer
try:
    # transformers < 5
    from transformers.modeling_utils import no_init_weights
except ImportError:
    # transformers >= 5 moved initialization helpers out of modeling_utils.
    from transformers.initialization import no_init_weights
from transformers.trainer_pt_utils import LabelSmoother

from ...auto.auto_config import AutoConfig
from ...auto.auto_model import AutoModel
from .model_config import LalmConfig
from .utils import (
    preprocess_text_and_audio_impl,
    preprocess_text_and_audio_packed_sdpa,
    replace_whisper_encoder_forward,
)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

CHAT_TEMPLATE = """{% for message in messages -%}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor -%}
{% if add_generation_prompt -%}
<|im_start|>assistant
{% endif -%}"""


def get_llm_hidden_size(llm_config) -> int:
    """Return the causal text width, including for a Qwen3-Omni Thinker."""
    # === QWEN3-OMNI-30B INTEGRATION START ===
    if hasattr(llm_config, "thinker_config"):
        llm_config = llm_config.thinker_config
    if hasattr(llm_config, "text_config"):
        llm_config = llm_config.text_config
    # === QWEN3-OMNI-30B INTEGRATION END ===
    return int(llm_config.hidden_size)


class EncoderProjector(nn.Module):
    def __init__(self, encoder_dim: int, llm_dim: int, downsample_rate: int = 5):
        super().__init__()
        self.downsample_rate = downsample_rate
        self.linear1 = nn.Linear(encoder_dim * self.downsample_rate, llm_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feat_dim = x.size()
        discard = seq_len % self.downsample_rate
        if discard > 0:
            x = x[:, :-discard, :]
        x = x.contiguous().view(
            batch_size, seq_len // self.downsample_rate, feat_dim * self.downsample_rate
        )
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class LalmModel(nn.Module):
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        map_location: str | torch.device = "cpu",
        llm_dtype: torch.dtype | str | None = torch.float16,
    ) -> "LalmModel":
        # Support HF Hub repo IDs
        if not os.path.exists(model_path):
            model_path = AutoModel._download_from_hub(model_path)
        # Resolve directory and checkpoint
        if os.path.isdir(model_path):
            model_dir = model_path
            weight_path = None
            for ext in (".safetensors", ".pt"):
                for name in ("pretrained", "model"):
                    p = os.path.join(model_dir, f"{name}{ext}")
                    if os.path.exists(p):
                        weight_path = p
                        break
                if weight_path is not None:
                    break
        else:
            model_dir, _ = os.path.split(model_path)
            weight_path = model_path

        # Build config & tokenizer
        config = AutoConfig.from_pretrained(model_dir)
        tokenizer = HFTokenizer.from_pretrained(model_dir, use_fast=False)

        # Instantiate model skeleton, allowing caller to control LLM dtype
        model = cls(config, tokenizer, llm_dtype=llm_dtype)

        # Load composite weights if present (may exclude some modules)
        if weight_path and os.path.exists(weight_path):
            ext = os.path.splitext(weight_path)[1].lower()
            if ext == ".safetensors":
                from safetensors.torch import load_file as safe_load_file

                device_arg = (
                    str(map_location)
                    if isinstance(map_location, torch.device)
                    else map_location
                )
                state_obj = safe_load_file(weight_path, device=device_arg)
            else:
                state_obj = torch.load(weight_path, map_location=map_location)
            state_dict = (
                state_obj.get("state_dict", state_obj)
                if isinstance(state_obj, dict)
                else state_obj
            )
            model.load_state_dict(state_dict, strict=False)

        # Optionally load excluded submodules saved separately under model_dir
        # 1) Audio encoder
        audio_subdir = os.path.join(model_dir, "audio_encoder")
        if os.path.isdir(audio_subdir):
            try:
                if "whisper" in model.audio_encoder_type:
                    # Prefer full model then extract encoder to match keys
                    from transformers.models.whisper.modeling_whisper import (
                        WhisperEncoder,
                    )

                    ae = WhisperEncoder.from_pretrained(audio_subdir)
                    model.audio_encoder.load_state_dict(ae.state_dict(), strict=True)
                else:
                    # Auden zipformer
                    from ...models.zipformer.model import ZipformerEncoderModel

                    ae = ZipformerEncoderModel.from_pretrained(audio_subdir)
                    model.audio_encoder.load_state_dict(ae.state_dict(), strict=True)
                logging.info(
                    f"[lalm.from_pretrained] Loaded audio encoder from {audio_subdir}"
                )
            except Exception as e:
                logging.warning(
                    f"[lalm.from_pretrained] Failed to load audio encoder from {audio_subdir}: {e}"
                )

        # 2) LLM (full HF model dir)
        llm_subdir = os.path.join(model_dir, "llm")
        if os.path.isdir(llm_subdir):
            try:
                loaded_llm = AutoModelForCausalLM.from_pretrained(
                    llm_subdir,
                    torch_dtype=model.llm_dtype,
                )
                model.llm = loaded_llm
                logging.info(f"[lalm.from_pretrained] Loaded LLM from {llm_subdir}")
            except Exception as e:
                logging.warning(
                    f"[lalm.from_pretrained] Failed to load LLM from {llm_subdir}: {e}"
                )

        model.eval()
        return model

    def __init__(
        self,
        config: LalmConfig,
        tokenizer,
        llm_dtype: torch.dtype | str | None = torch.float16,
        pretrained_llm=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.audio_encoder_config = config.audio_encoder_config
        self.llm_config = config.llm_config
        self.tokenizer = tokenizer
        self.use_flash_attn = config.use_flash_attn
        self.audio_encoder_type = self.audio_encoder_config.model_type
        assert self.audio_encoder_type in ["zipformer", "whisper"]

        # Decide LLM dtype (default: float16). Accept both torch.dtype and common string aliases.
        if isinstance(llm_dtype, str):
            name = llm_dtype.lower()
            if name in ("float16", "fp16", "half"):
                self.llm_dtype: torch.dtype = torch.float16
            elif name in ("bfloat16", "bf16"):
                self.llm_dtype = torch.bfloat16
            elif name in ("float32", "fp32"):
                self.llm_dtype = torch.float32
            else:
                logging.warning(
                    f"[LalmModel] Unknown llm_dtype='{llm_dtype}', defaulting to float16."
                )
                self.llm_dtype = torch.float16
        elif isinstance(llm_dtype, torch.dtype):
            self.llm_dtype = llm_dtype
        else:
            # None or unsupported type → default to float16
            self.llm_dtype = torch.float16

        self.exclude_from_checkpoint = config.exclude_from_checkpoint
        if self.exclude_from_checkpoint:
            logging.info(f"Exclude from checkpoints: {self.exclude_from_checkpoint}")

        self.pad_token_id = self.tokenizer.pad_token_id
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(
            self.config.audio_token
        )

        # 1) Audio encoder
        if self.audio_encoder_type == "whisper":
            from transformers.models.whisper.modeling_whisper import WhisperEncoder

            replace_whisper_encoder_forward()
            with no_init_weights():
                self.audio_encoder = WhisperEncoder(self.audio_encoder_config)
            self.audio_encoder_dim = self.audio_encoder_config.d_model
        else:
            self.audio_encoder = AutoModel.from_config(self.config.audio_encoder_config)
            self.audio_encoder_dim = self.audio_encoder.encoder_out_dim

        # 2) LLM
        attn_impl = "flash_attention_2" if self.use_flash_attn else "sdpa"
        # === QWEN3-OMNI-30B INTEGRATION START ===
        # Accepting the already-loaded Thinker avoids holding two 30B model
        # instances while TagSpeech is assembled.
        if pretrained_llm is not None:
            self.llm = pretrained_llm
        else:
            with no_init_weights():
                if self.config.llm_config.model_type == "qwen3_omni_moe_thinker":
                    from transformers import Qwen3OmniMoeThinkerForConditionalGeneration

                    self.llm = Qwen3OmniMoeThinkerForConditionalGeneration(
                        self.config.llm_config
                    )
                else:
                    self.llm = AutoModelForCausalLM.from_config(
                        self.config.llm_config,
                        attn_implementation=attn_impl,
                        torch_dtype=self.llm_dtype,
                    )
        # === QWEN3-OMNI-30B INTEGRATION END ===

        # 3) Projector
        self.encoder_projector = EncoderProjector(
            self.audio_encoder_dim,
            get_llm_hidden_size(self.llm.config),
            self.config.audio_encoder_projector_ds_rate,
        )
        if self.config.tag_audio_boundary:
            self.audio_tag_embedding = nn.Parameter(
                torch.zeros((2, get_llm_hidden_size(self.llm.config)), dtype=self.llm_dtype)
            )

    def forward_audio_features(self, x: torch.Tensor, x_lens: torch.Tensor):
        if "whisper" in self.audio_encoder_type:
            x = x.transpose(1, 2)
            encoder_outs = self.audio_encoder(x)[0]
            feature_lens = (x_lens - 1) // 2 + 1
        else:
            encoder_output = self.audio_encoder(x, x_lens)
            encoder_outs = encoder_output.encoder_out
            feature_lens = encoder_output.encoder_out_lens

        audio_features = self.encoder_projector(encoder_outs).to(self.llm_dtype)
        feature_lens = feature_lens // self.config.audio_encoder_projector_ds_rate
        return audio_features, feature_lens

    def preprocess_text_and_audio(
        self,
        messages: List[List[Dict[str, str]]],
        audio_features: Optional[torch.Tensor],
        audio_feature_lens: Optional[torch.Tensor],
        max_length: int = 256,
        tag_audio_boundary: bool = False,
        is_training: bool = False,
        pack_sequences: bool = True,
        max_total_length: Optional[int] = None,
    ):
        # Choose preprocessing path: packed (isolated) or original
        if pack_sequences and not self.use_flash_attn:
            return preprocess_text_and_audio_packed_sdpa(
                messages=messages,
                tokenizer=self.tokenizer,
                llm=self.llm,
                audio_token=self.config.audio_token,
                audio_features=audio_features,
                audio_feature_lens=audio_feature_lens,
                max_length=max_length,
                is_training=is_training,
                chat_template=CHAT_TEMPLATE,
                max_total_length=max_total_length,
            )
        # Original implementation
        return preprocess_text_and_audio_impl(
            messages=messages,
            tokenizer=self.tokenizer,
            llm=self.llm,
            audio_token=self.config.audio_token,
            audio_features=audio_features,
            audio_feature_lens=audio_feature_lens,
            max_length=max_length,
            tag_audio_boundary=tag_audio_boundary,
            is_training=is_training,
            audio_tag_embedding=getattr(self, "audio_tag_embedding", None),
            chat_template=CHAT_TEMPLATE,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        messages: List[List[Dict[str, str]]],
        max_length: Optional[int] = None,
        pack_sequences: bool = True,
        max_total_length: Optional[int] = None,
    ):
        audio_features, feature_lens = self.forward_audio_features(x, x_lens)
        input_ids, inputs_embeds, attention_mask, labels, position_ids = (
            self.preprocess_text_and_audio(
                messages,
                audio_features=audio_features,
                audio_feature_lens=feature_lens,
                max_length=max_length,
                is_training=True,
                tag_audio_boundary=self.config.tag_audio_boundary,
                pack_sequences=pack_sequences,
                max_total_length=max_total_length,
            )
        )
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )
        with torch.no_grad():
            preds = torch.argmax(outputs.logits, -1)
            acc = compute_accuracy(
                preds[:, :-1],
                labels[:, 1:],
                ignore_label=IGNORE_TOKEN_ID,
            )
        return outputs, acc

    def generate(
        self, input, messages, max_length=None, pack_sequences=False, **kwargs
    ):
        if input is not None:
            if isinstance(input, tuple) and len(input) == 2:
                x, x_lens = input
            else:
                x, x_lens = self.audio_encoder.extract_feature(input)
            audio_features, feature_lens = self.forward_audio_features(x, x_lens)

        # enforce left padding for batched generation
        self.tokenizer.padding_side = "left"
        max_total_length = getattr(self.config, "max_total_length", None)
        input_ids, inputs_embeds, attention_mask, _, _ = self.preprocess_text_and_audio(
            messages,
            audio_features=audio_features,
            audio_feature_lens=feature_lens,
            max_length=max_length,
            is_training=False,
            tag_audio_boundary=self.config.tag_audio_boundary,
            pack_sequences=pack_sequences,
            max_total_length=max_total_length,
        )
        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            bos_token_id=self.llm.config.bos_token_id,
            eos_token_id=self.llm.config.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True,
            **kwargs,
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def compute_accuracy(
    pad_outputs: torch.Tensor,
    pad_targets: torch.Tensor,
    ignore_label: int,
):
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_outputs.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    return numerator.float() / denominator.float()
