import logging
import json
import os
from typing import List

import torch
import torch.nn as nn

from ...auto.auto_config import AutoConfig
from ...auto.auto_model import AutoModel
from .features import construct_feature_extractor as _construct_fbank
from .features import extract_features as _extract_features
from .modules.subsampling import Conv2dSubsampling
from .modules.zipformer import Zipformer2
from .utils.padding import make_pad_mask
from .utils.scaling import ScheduledFloat


class ZipformerEncoderModel(nn.Module):
    """
    This is the base zipformer encoder model
    """

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        *,
        module_name=None,
        strict: bool = True,
        map_location="cpu",
        dtype=None,
        device=None,
    ):
        """Load a Zipformer encoder (or extract it from a composite model).

        Args:
            model_path: Directory containing weights and config, or a direct path to a .pt file.
            module_name: If loading from a composite model, the submodule name containing
                the encoder weights (e.g., "audio_encoder" or "encoder"). If None, it will
                be auto-detected based on the config.
            strict: Passed to ``load_state_dict``.
            map_location: Passed to ``torch.load``.
            dtype: Optional dtype to move the model to after loading.
            device: Optional device to move the model to after loading.
        """
        # Support HF Hub repo IDs
        if not os.path.exists(model_path):
            model_path = AutoModel._download_from_hub(model_path)
        # Resolve model_dir and weight file
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
            if weight_path is None:
                raise FileNotFoundError(
                    f"Expected one of ['pretrained.safetensors','model.safetensors','pretrained.pt','model.pt'] under {model_dir}"
                )
        else:
            weight_path = model_path
            model_dir, _ = os.path.split(model_path)

        # A converted encoder may be stored as a named weight/config pair, e.g.
        # model_ami_audio.safetensors + config_ami_audio.json.  When a direct
        # weight path is supplied, prefer its matching config instead of the
        # unrelated config.json that may also exist in the directory.
        config_path = None
        if not os.path.isdir(model_path):
            weight_stem = os.path.splitext(os.path.basename(weight_path))[0]
            if weight_stem.startswith("model_"):
                suffix = weight_stem[len("model_") :]
                candidate = os.path.join(model_dir, f"config_{suffix}.json")
                if os.path.exists(candidate):
                    config_path = candidate

        if config_path is not None:
            with open(config_path, "r") as f:
                config = AutoConfig.for_model(**json.load(f))
            logging.info(f"Loaded encoder config from {config_path}")
        else:
            config = AutoConfig.from_pretrained(model_dir)
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
        if isinstance(state_obj, dict) and "state_dict" in state_obj:
            state_dict = state_obj["state_dict"]
        else:
            state_dict = state_obj
        logging.info(f"Loaded weights for type {config.model_type} from {weight_path}")

        # Determine sub-config: allow both composite configs and direct encoder config
        detected_module = None
        if module_name is not None:
            detected_module = module_name
        elif hasattr(config, "audio_encoder_config"):
            detected_module = "audio_encoder"
        elif hasattr(config, "speech_encoder_config"):
            detected_module = "speech_encoder"
        elif hasattr(config, "encoder_config"):
            detected_module = "encoder"

        if detected_module is not None:
            sub_config = getattr(config, f"{detected_module}_config", None)
            if sub_config is None:
                raise ValueError(
                    f"Config does not contain '{detected_module}_config' needed for encoder."
                )
        else:
            # Direct encoder config saved at model_dir
            sub_config = config

        model = cls(sub_config)

        # Strip possible prefixes in weights: DDP 'module.', and parent module prefix
        def _strip_prefix(d: dict, prefix: str):
            if any(k.startswith(prefix) for k in d.keys()):
                return {
                    k[len(prefix) :]: v for k, v in d.items() if k.startswith(prefix)
                }
            return None

        candidate = state_dict
        # Strip DDP prefix at most once
        stripped = _strip_prefix(candidate, "module.")
        if stripped:
            candidate = stripped
        # For composite parents, strip one parent-level prefix; skip for direct encoders
        if detected_module is not None:
            for pfx in [
                f"{detected_module}.",
                "audio_encoder.",
                "speech_encoder.",
                "encoder.",
            ]:
                stripped = _strip_prefix(candidate, pfx)
                if stripped:
                    candidate = stripped
                    break
        state_dict = candidate

        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if not strict and (missing or unexpected):
            logging.warning(
                f"from_pretrained loaded with missing keys: {missing}, unexpected keys: {unexpected}"
            )

        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device)
        model.eval()
        return model

    def __init__(self, config, init_batch_count=0):
        super().__init__()
        self.config = config
        self.feature_extractor = None
        # Initialize encoder embedding
        self.encoder_embed = Conv2dSubsampling(
            in_channels=config.feature_dim,
            out_channels=config.encoder_dim[0],
            dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        )
        # Initialize encoder
        self.encoder = Zipformer2(
            output_downsampling_factor=config.output_downsampling_factor,
            downsampling_factor=tuple(config.downsampling_factor),
            encoder_dim=config.encoder_dim,
            num_encoder_layers=config.num_encoder_layers,
            encoder_unmasked_dim=config.encoder_unmasked_dim,
            query_head_dim=config.query_head_dim,
            pos_head_dim=config.pos_head_dim,
            value_head_dim=config.value_head_dim,
            num_heads=config.num_heads,
            feedforward_dim=config.feedforward_dim,
            cnn_module_kernel=config.cnn_module_kernel,
            pos_dim=config.pos_dim,
            dropout=config.dropout,
            warmup_batches=config.warmup_batches,
            causal=config.causal,
            chunk_size=tuple(config.chunk_size),
            left_context_frames=tuple(config.left_context_frames),
        )
        self.encoder_out_dim = max(config.encoder_dim)
        # model_batch_count for ScheduledFloat
        self.register_buffer(
            "model_batch_count",
            torch.tensor(init_batch_count, dtype=torch.float),
            persistent=True,
        )

    def set_batch_count(self):
        for m in self.modules():
            if hasattr(m, "batch_count"):
                m.batch_count = float(self.model_batch_count.item())

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Backward-compatible loading: ensure 'model_batch_count' exists.

        Old checkpoints lack this registered buffer. We set it to 100000
        so strict=True loads remain possible without warnings and ScheduledFloat is saturated.
        """
        key = prefix + "model_batch_count"
        if key not in state_dict:
            # Default to a large value (100000) to saturate ScheduledFloat for old ckpts
            state_dict[key] = torch.tensor(
                100000.0,
                dtype=self.model_batch_count.dtype,
                device=self.model_batch_count.device,
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor, return_dict: bool = True
    ) -> dict | tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
        Args:
            x: A 3-D tensor of shape (N, T, C).
            x_lens: A 1-D tensor of shape (N,), number of valid frames per sample.
            return_dict: If True, return a dict with keys
                'encoder_out', 'encoder_out_lens', 'encoder_out_full'.
                If False, return a tuple (encoder_out, encoder_out_lens, encoder_out_full).
        """
        x, x_lens = self.encoder_embed(x, x_lens)
        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        outputs = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = outputs[0].permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        encoder_out_lens = outputs[1]
        encoder_out_full = outputs[2].permute(0, 2, 1, 3)  # (N_blocks, N, T, C)

        if return_dict:
            return {
                "encoder_out": encoder_out,
                "encoder_out_lens": encoder_out_lens,
                "encoder_out_full": encoder_out_full,
            }
        else:
            return encoder_out, encoder_out_lens, encoder_out_full

    def save_pretrained(
        self,
        save_directory: str,
        *,
        filename: str | None = None,
        use_safetensors: bool = True,
    ) -> str:
        """Save model weights and config to a directory (HF-style).

        Writes config via ``config.save_pretrained(save_directory)`` and weights to
        ``model.safetensors`` by default (falls back to ``model.pt`` if
        safetensors is unavailable or ``use_safetensors=False``).

        Args:
            save_directory: Target directory.
            filename: Optional explicit filename for weights. If None, chooses
                ``model.safetensors`` or ``model.pt``.
            use_safetensors: Prefer the safetensors format when possible.

        Returns:
            The path of the saved weight file.
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save configuration next to weights
        # Assumes encoder sub-config supports save_pretrained
        self.config.save_pretrained(save_directory)

        # Decide filename and format
        weight_path: str
        chosen_filename = filename
        if chosen_filename is None:
            if use_safetensors:
                try:
                    from safetensors.torch import (  # noqa: F401
                        save_file as safe_save_file,
                    )

                    chosen_filename = "model.safetensors"
                except Exception:
                    chosen_filename = "model.pt"
            else:
                chosen_filename = "model.pt"

        weight_path = os.path.join(save_directory, chosen_filename)

        # Always save CPU state_dict for portability
        state_dict = {k: v.detach().cpu() for k, v in self.state_dict().items()}

        ext = os.path.splitext(weight_path)[1].lower()
        if ext == ".safetensors":
            try:
                from safetensors.torch import save_file as safe_save_file

                safe_save_file(state_dict, weight_path)
            except Exception as e:
                # Fallback to .pt if safetensors is unavailable
                logging.warning(
                    f"Failed to save safetensors ({e}); falling back to PyTorch .pt"
                )
                weight_path = os.path.splitext(weight_path)[0] + ".pt"
                torch.save(state_dict, weight_path)
        else:
            torch.save(state_dict, weight_path)

        logging.info(f"Saved model to {weight_path}")
        return weight_path

    def construct_feature_extractor(
        self,
        *,
        sample_rate: int = 16000,
        num_mel_bins: int = 80,
        device: str | torch.device = "cpu",
    ):
        """Construct a default FBANK feature extractor (kaldifeat).

        This is a thin wrapper delegating to ``auden.models.zipformer.features``.
        """
        return _construct_fbank(
            sample_rate=sample_rate, num_mel_bins=num_mel_bins, device=device
        )

    def extract_feature(self, input):
        """Thin wrapper to extract features via the helper utility.

        Accepts the same input forms as before and returns (features, feature_lens).
        """
        if not self.feature_extractor:
            self.feature_extractor = self.construct_feature_extractor()
        device = next(self.encoder.parameters()).device
        return _extract_features(
            input, self.feature_extractor, target_sample_rate=16000, device=device
        )
