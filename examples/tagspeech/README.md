# Qwen2.5 Omni with sinusoidal speaker kernels

This TagSpeech variant performs multi-speaker transcription and diarization using a frozen Qwen2.5 Omni Thinker, separate semantic and voice encoders, and trainable speaker conditioning.

The main pipeline is:

```text
Audio → semantic encoder → projector → numeric anchors ────── queries
Audio → voice encoder → projector → temporal convolution
                                  → sinusoidal kernel ────── keys/values
                                            │                    │
                                            ↓                    ↓
                                      boundary head       cross-attention
                                                                 ↓
                                                            MLP adapter
                                                                 ↓
                                               residual to semantic features
                                                                 ↓
                                                     Qwen2.5 Omni Thinker
                                                                 ↓
                                                  timestamped text and speakers
```

The kernel operates on speaker features. Cross-attention combines the streams afterward. Temporal convolution also uses a residual connection. The main configuration trains with XML token cross-entropy plus boundary binary cross-entropy, with both loss weights set to 1.0. It does not use CTC.

`ordered_kernel_*` is the existing configuration naming for sinusoidal kernels. “Ordered” refers to speaker slots indexed by first appearance within an input example. These keys remain unchanged for compatibility. The LLM generates speaker IDs; kernel slots are not directly decoded into XML IDs.



## Setup

From the repository root, install Auden in an environment with compatible PyTorch, Torchaudio, k2, and Transformers dependencies:

```bash
pip install -e .
pip install meeteval scipy
python -c "from transformers import Qwen2_5OmniThinkerForConditionalGeneration"
cd examples/tagspeech
```

The import check verifies that the installed Transformers version exposes the required Thinker class. Training requires GPU resources appropriate for the frozen 7B model and the configured batch duration.

## Set local paths

Run the following commands from `examples/tagspeech`. Paths beginning with `/path/to/` are placeholders and must be replaced. Relative paths are resolved from this directory.

| Asset | Setting or location |
|---|---|
| Complete local Qwen2.5 Omni checkpoint | `model.llm.pretrained_model` in `configs/train_qwen25_omni_7b_speaker_kernel_alimeeting.yaml`; inherited by AMI |
| Semantic encoder checkpoint | `model.audio_encoder.pretrained_model` in the selected dataset config |
| Voice encoder checkpoint | `model.voice_encoder.pretrained_model` in the selected dataset config |
| Train/validation/test manifests | `manifest` entries in `configs/AMI/data_configs/` or `configs/AliMeeting/data_configs/` |
| Training output directory | `exp_dir` in the selected training config |
| Model directory for decoding | `exp_dir` in `configs/decode.yaml` |

The Omni loader uses local files only. Set its path to a complete downloaded checkpoint directory, not just a weights file. For example:

```yaml
model:
  llm:
    pretrained_model: /path/to/Qwen2.5-Omni-7B
```

The AMI defaults expect encoder files under `experiments/diarization_AMI/audio_encoder/` and `experiments/diarization_AMI/voice_encoder/`. Place the corresponding encoder weights and configurations there, or update their paths. The older `diarization_AMI` directory supplies encoder assets; the new training output uses a separate directory.

Dataset manifests must reference accessible audio files and, when applicable, feature files. Updating the YAML manifest location alone does not rewrite paths inside a manifest. Prepare manifests with recording paths and speaker-annotated supervisions before training.

## Training

For AMI:

```bash
bash scripts/train_AMI_qwen25_omni_7b_crossattn.sh \
  model.llm.pretrained_model=/path/to/Qwen2.5-Omni-7B
```

For AliMeeting:

```bash
bash scripts/train_AliMeeting_qwen25_omni_7b_crossattn.sh \
  model.llm.pretrained_model=/path/to/Qwen2.5-Omni-7B
```

These scripts launch one process and accept additional Hydra overrides. AMI inherits the AliMeeting cross-attention configuration, which inherits `configs/train.yaml`; retain all three files. Numeric anchor embeddings are generated from the active Omni decoder during training and saved with the experiment. No separate anchor-generation command is needed for this configuration.

The provided configuration enables temporal convolution, sinusoidal speaker kernels, cross-attention with an MLP adapter, and boundary supervision.

## Decoding

`configs/decode.yaml` selects:

```yaml
exp_dir: experiments/diarization_AMI_speaker_kernel_crossattn_qwen25_omni_7b
```

Set `checkpoint.iter` to an iteration present in that directory and `checkpoint.avg` to the desired number of available checkpoints to average. The checked-in iteration is an example, not a selected paper checkpoint.

```bash
# Example: use only if these checkpoints are available.
python decode.py checkpoint.iter=28000 checkpoint.avg=5
```

Alternatively, select an existing checkpoint by filename:

```bash
python decode.py checkpoint.filename=checkpoint-28000.pt
```

`checkpoint.filename` takes precedence over iteration/epoch averaging. For AliMeeting, also override `exp_dir` and `data.test_data_config` with the corresponding AliMeeting paths.

When moving a trained experiment to another machine, update `qwen2_5_omni_pretrained_model` in its saved `config.json`. Keep its tokenizer files, `digit_embeddings.pt`, encoder configurations and weights, and selected model/trainer checkpoints available. Saved experiment files are separate from the source YAML configuration.

## WAV inference

```bash
python inference.py /path/to/audio.wav \
  --model /path/to/exported-model
```

The exported model directory must contain `model.pt` or `model.safetensors` plus its config, tokenizer, numeric embeddings, and required encoder assets. Use `decode.py` for trainer-checkpoint averaging and manifest-based evaluation.

## Output and evaluation

Decoding writes reference and predicted XML-style transcripts under `<exp_dir>/greedy_search/`. Output includes timestamps, text, speaker IDs, and gender tags:

```xml
<text>
0.00-1.50>Hello.
1.50-3.00>Hi there.
</text>
<speaker>
<spk id="1" g="m" t="0.00-1.50"/>
<spk id="2" g="f" t="1.50-3.00"/>
</speaker>
```

Evaluate a generated results file:

```bash
python evaluate.py --xml_file /path/to/xml-outputs-ami-test-utterance-group-iter-28000-avg-5.txt
```

## Implementation

| File | Purpose |
|---|---|
| `train.py` / `model_config.py` | Model loading and configuration |
| `model.py` | Encoders, projectors, numeric anchors, and LLM integration |
| `modules/speaker_kernel_cross_attention.py` | Temporal convolution, cross-attention, MLP, and boundary head |
| `modules/speaker_only_kernel.py` | Speaker probabilities and sinusoidal kernel encoding |
| `trainer.py` | XML/LLM and boundary losses |
| `data_module.py` / `multi_speaker_dataset.py` | Manifest loading and batching |
| `decode.py` / `evaluate.py` | Decoding and evaluation |

The current imports also require the legacy `shared_segment_conditioner.py` and `sin_kernel_ordered.py` files, even though the main cross-attention configuration does not execute their conditioner. Include `src/auden/` and the package installation files when sharing the implementation.

## License and attribution

This implementation builds on [TagSpeech in Auden](https://github.com/AudenAI/Auden/tree/main/examples/tagspeech). The repository's [LICENSE](../../LICENSE) and [NOTICE](../../NOTICE) accompany this code.

## TagSpeech citation

```bibtex
@article{huo2026tagspeech,
  title={TagSpeech: End-to-End Multi-Speaker ASR and Diarization with Fine-Grained Temporal Grounding},
  author={Huo, Mingyue and Shao, Yiwen and Zhang, Yuheng},
  journal={arXiv preprint arXiv:2601.06896},
  year={2026}
}
```
