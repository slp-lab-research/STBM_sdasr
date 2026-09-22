"""TagSpeech training loop built on BaseTrainer.

This trainer wires the TagSpeech model with the Lhotse-based datamodule and logs
token-level loss and accuracy on validation. See examples/tagspeech/configs for details.
"""

import logging
import random

import torch
import torch.nn.functional as F
from utils.xml_utils import construct_multi_speaker_xml

from auden.trainer.ddp_trainer import BaseTrainer
from auden.utils.metric_tracker import MetricsTracker


class TagSpeechTrainer(BaseTrainer):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        prompt_file = cfg.prompt_file
        with open(prompt_file, "r", encoding="utf-8") as f:
            self.prompt_list = [line.strip() for line in f if line.strip()]

        # Cache for XML generation to avoid recomputation
        self._xml_cache = {}

        # Log max_length configuration
        max_length = getattr(self.model.config, "max_length", 800)
        logging.info(f"[TagSpeechTrainer] Model max_length: {max_length}")

    def _forward_one_batch(self, batch: dict, is_training: bool, return_emb=False):
        device = self.device
        feature = batch["inputs"]

        # Check feature dimensions: fbank (N, T, C)
        assert feature.ndim == 3, f"Expected fbank (B, T, C), got shape {feature.shape}"

        feature = feature.to(device)

        supervisions = batch["supervisions"]
        feature_lens = supervisions["num_frames"].to(device)

        # Get cuts for multi-speaker processing
        cuts = batch["supervisions"]["cut"]
        audio_token = self.cfg.model.audio_token
        batch_size = len(cuts)
        messages = []

        for i, cut in enumerate(cuts):
            # Kernel pipeline: single-stream audio prompt.
            if self.cfg.model.get("use_segment_speaker_kernel", False):
                user_content = f"<audio>{audio_token}</audio>"
            else:
                user_content = (
                    f"<text>{audio_token}</text>\n"
                    f"<speaker>{audio_token}</speaker>"
                )

            # Construct multi-speaker XML target from cut supervisions (with caching)
            target_xml = self._construct_multi_speaker_xml(cut)

            message = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": target_xml},
            ]
            messages.append(message)
        with torch.set_grad_enabled(is_training):
            # Get max_length from config if available
            max_length = getattr(self.model.config, "max_length", 800)
            # Kernel, MLP, and temporal convolution run inside the model.
            model_outputs, acc = self.model(
                x=feature,
                x_lens=feature_lens,
                messages=messages,
                max_length=max_length,
            )
            # XML/LLM loss: token cross-entropy.
            shift_logits = model_outputs.logits[:, :-1, :].contiguous()
            training_labels = model_outputs.get("training_labels")
            shift_labels = (
                training_labels[:, 1:].contiguous()
                if training_labels is not None
                else None
            )
            if shift_labels is None:
                # Hugging Face causal-LM outputs do not retain input labels.
                llm_loss = model_outputs.loss
            else:
                llm_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            # Boundary loss: masked binary cross-entropy.
            boundary_logits = model_outputs.get("boundary_logits")
            segment_padding_mask = model_outputs.get("segment_padding_mask")
            use_boundary_loss = bool(
                self.cfg.model.get("use_boundary_loss", False)
            )
            if use_boundary_loss:
                if boundary_logits is None or segment_padding_mask is None:
                    raise RuntimeError(
                        "Boundary loss is enabled but boundary outputs are missing"
                    )
                boundary_targets = self._build_speaker_change_targets(
                    cuts=cuts,
                    segment_lens=model_outputs["segment_lens"],
                    max_segments=boundary_logits.size(1),
                    device=boundary_logits.device,
                )
                valid_boundary_mask = ~segment_padding_mask
                boundary_loss = F.binary_cross_entropy_with_logits(
                    boundary_logits[valid_boundary_mask],
                    boundary_targets[valid_boundary_mask],
                )
            else:
                # No boundary contribution when disabled.
                boundary_loss = llm_loss.new_zeros(())
            # Total loss: weighted XML/LLM and boundary losses.
            llm_weight = float(
                self.cfg.trainer.get("llm_loss_weight", 1.0)
            )
            boundary_weight = float(
                self.cfg.trainer.get("boundary_loss_weight", 1.0)
            )
            loss = llm_weight * llm_loss + boundary_weight * boundary_loss

        assert loss.requires_grad == is_training

        # Batch metrics.
        info = MetricsTracker()
        num_frames = sum(len(text) for text in messages)
        info.set_value("frames", num_frames, normalization="sum")
        info.set_value("samples", batch_size, normalization="sum")
        # Total weighted loss.
        info.set_value("loss", loss.detach().cpu().item(), normalization="frame_avg")
        # XML token prediction loss.
        info.set_value(
            "llm_loss",
            llm_loss.detach().cpu().item(),
            normalization="frame_avg",
        )
        # Alias of llm_loss; same value.
        info.set_value(
            "xml_loss",
            llm_loss.detach().cpu().item(),
            normalization="frame_avg",
        )
        # Unweighted boundary loss.
        info.set_value(
            "boundary_loss",
            boundary_loss.detach().cpu().item(),
            normalization="frame_avg",
        )
        # Token prediction accuracy.
        info.set_value("acc", acc, normalization="sample_avg")

        # Explicit cleanup to prevent memory leaks
        del feature, feature_lens, messages, model_outputs
        if not is_training and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss, info


    # Boundary targets: mark speaker changes.
    @staticmethod
    def _build_speaker_change_targets(
        cuts,
        segment_lens: torch.Tensor,
        max_segments: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Map chronological speaker changes onto the shared segment grid."""
        targets = torch.zeros(
            len(cuts), max_segments, dtype=torch.float32, device=device
        )
        for batch_idx, cut in enumerate(cuts):
            num_segments = int(segment_lens[batch_idx].item())
            duration = float(cut.duration)
            if num_segments <= 0 or duration <= 0:
                continue

            supervisions = sorted(
                cut.supervisions,
                key=lambda supervision: (
                    float(supervision.start),
                    float(supervision.duration),
                ),
            )
            previous_speaker = None
            for supervision in supervisions:
                speaker = supervision.speaker
                if (
                    previous_speaker is not None
                    and speaker is not None
                    and speaker != previous_speaker
                ):
                    relative_start = min(
                        max(float(supervision.start), 0.0), duration
                    )
                    segment_idx = min(
                        int(relative_start / duration * num_segments),
                        num_segments - 1,
                    )
                    targets[batch_idx, segment_idx] = 1.0
                if speaker is not None:
                    previous_speaker = speaker
        return targets

    def _construct_multi_speaker_xml(self, cut):
        """Construct XML format target from cut with multiple supervisions.

        Uses the configured XML format version.

        Args:
            cut: Lhotse Cut object with multiple supervisions

        Returns:
            str: XML formatted multi-speaker transcript
        """
        # Check cache first
        cut_id = cut.id
        if cut_id in self._xml_cache:
            return self._xml_cache[cut_id]

        # Use XML constructor
        result = construct_multi_speaker_xml(cut)

        # Cache the result
        self._xml_cache[cut_id] = result

        return result

    def validate(self, epoch: int):
        """
        Validation is provided by BaseTrainer.

        Override in a subclass if you need TagSpeech specific validation logic
        (e.g., generation-based metrics like WER/CER).
        """
        return super().validate(epoch)
