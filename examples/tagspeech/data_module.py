"""
Note: Unlike ASR tasks that only use the first supervision, this diarization implementation **preserves all supervisions** in each cut to maintain multi-speaker information. The `data_module.py` is specifically designed for multi-speaker scenarios and cannot be replaced with the ASR version.

ASR example datamodule built on top of BaseLhotseDatamodule.

This module provides a task-specific implementation for ASR training/validation
using Lhotse CutSet and k2's K2SpeechRecognitionDataset. It focuses on:

- Building a mixed training CutSet from multiple manifests with per-dataset
  hour-based weights.
- Dataset-level text normalization and basic filtering by duration and text
  length.
- Optional padding to 30s (useful for Whisper-style encoders).
- Dynamic bucketing for validation and a customizable training sampler inherited
  from BaseLhotseDatamodule.

Expected config keys (subset):
- cfg.train_data_config, cfg.valid_data_config, cfg.test_data_config (str):
  file paths to YAMLs describing dataset lists.
- cfg.text_normalization (bool): whether to normalize transcripts.
- cfg.pad_to_30s (bool): whether to pad all audio to 30s.
- cfg.use_infinite_dataset (bool): whether to repeat datasets infinitely.
- cfg.sampler.max_duration (float): max duration used by bucketing sampler.

YAML schema:
- Train (examples/asr/configs/data_configs/train_data_config.yaml)
  - A list of entries with keys:
    - manifest (str): absolute path to Lhotse CutSet jsonl.gz
    - hours (float): approximate hours to weight sampling
    - weights (int, optional): repetition multiplier (default 1)
    - lang (str, optional): language tag (default "zh")
- Valid/Test (examples/asr/configs/data_configs/*_data_config.yaml)
  - A list of entries with keys:
    - name (str): short tag for logging/results
    - manifest (str): absolute path to Lhotse CutSet jsonl.gz

See BaseLhotseDatamodule for common fields like sampling_rate, transforms,
input_strategy, and sampler construction utilities.
"""

import logging
from collections import defaultdict
from functools import partial

import torch
import yaml
from lhotse import CutSet, set_audio_duration_mismatch_tolerance
from lhotse.dataset import DynamicBucketingSampler, K2SpeechRecognitionDataset
from multi_speaker_dataset import MultiSpeakerAsrDataset
from torch.utils.data import DataLoader

from auden.data.lhotse_datamodule import BaseLhotseDatamodule, _SeedWorkers
from auden.utils.text_normalization import text_normalization


def _select_channel(cut, channel: int):
    """Select one audio channel without making LazyMapper unpicklable."""
    return cut.with_channels(channel)


class AsrDatamodule(BaseLhotseDatamodule):
    """ASR datamodule specialized from BaseLhotseDatamodule.

    This class wires up training and validation pipelines for ASR tasks using
    Lhotse CutSet and k2 datasets. It mixes multiple training datasets, applies
    basic text/audio filtering, and builds PyTorch DataLoaders.
    """

    def __init__(self, cfg):
        """Initialize the datamodule.

        Notes
        -----
        Some public corpora contain small audio/manifest mismatches. We relax
        Lhotse's duration mismatch tolerance slightly to avoid interruptions.
        """
        # NOTE: some data contains minor inconsistency
        set_audio_duration_mismatch_tolerance(0.5)
        super().__init__(cfg)

    def _filter_cutset(self, cutset, split="train"):
        """Apply text normalization and basic filtering to a CutSet.

        Operations
        ----------
        - Optional text normalization (lowercase, remove symbols).
        - Keep utterances with duration in [1s, 30s].
        - If multiple supervisions exist, keep the first one.
        - Remove empty or abnormally long texts (relative to duration).
        - Optionally pad all audio to 30s (for Whisper-style encoders).

        Parameters
        ----------
        cutset : CutSet
            The input CutSet.
        split : str
            "train" or "valid"; informational only here.

        Returns
        -------
        CutSet
            The filtered (and possibly normalized/padded) CutSet.
        """

        def text_normalization_on_cut(c):
            # Apply text normalization to all supervisions
            for sup in c.supervisions:
                text = sup.text
                text = text_normalization(
                    text,
                    case="lower",
                    space_between_cjk=False,
                    remove_diacritics=False,
                    remove_symbols=True,
                )
                sup.text = text
            return c

        def remove_short_and_long_utt(c):
            # Keep only utterances with duration between 0.5 second and 30 seconds
            if c.duration < 0.5 or c.duration > 30.0:
                return False
            # For multi-speaker cuts, keep all supervisions (don't filter to first one)
            # Check if any supervision has empty or abnormally long text
            for sup in c.supervisions:
                text = sup.text
                if len(text) == 0 or len(text) > c.duration * 30:
                    return False

            # Additional check: if there are too many overlapping supervisions, skip this cut
            # This might cause audio feature extraction issues
            if len(c.supervisions) > 10:  # Skip cuts with too many supervisions
                return False

            return True

        # Select single channel if configured (for multi-channel audio files)
        # This must be done before text normalization and filtering to ensure
        # all subsequent operations work with mono audio
        channel = self.cfg.get("channel", None)
        if channel is not None:
            logging.info(f"Selecting channel {channel} for multi-channel audio")
            cutset = cutset.map(partial(_select_channel, channel=channel))

        if self.cfg.text_normalization:
            cutset = cutset.map(text_normalization_on_cut)
        cutset = cutset.filter(remove_short_and_long_utt)

        if self.cfg.get("pad_to_30s", False):
            logging.info(
                "Padded all audio to 30s. This is usually used with `Whisper` encoder."
            )
            cutset = cutset.pad(duration=30)

        return cutset

    def _build_train_mux_cutset(self, train_data_config):
        """Build a mixed training CutSet from multiple manifests.

        The provided configuration is a list of dataset specs, each with:
        - manifest (str): path to Lhotse CutSet jsonl.gz
        - hours (float): approximate dataset hours, used for weighting
        - weights (int, optional): repetition multiplier (default 1)
        - lang (str, optional): language label for logging (default "zh")

        If cfg.use_infinite_dataset is True, each CutSet is repeated infinitely.
        Otherwise, we repeat according to "weights" to avoid early exhaustion
        during muxing. We then mux CutSets with weights proportional to their
        (hours * weights) to balance sampling across datasets.

        Parameters
        ----------
        train_data_config : List[dict]
            Parsed from cfg.train_data_config (YAML).

        Returns
        -------
        CutSet
            A single mixed CutSet ready for training.
        """
        cutset_list = []
        cutset_hours = []
        langs_hours = defaultdict(int)

        for train_set in train_data_config:
            logging.info(f"Getting {train_set['manifest']} cuts")
            cutset = CutSet.from_file(train_set["manifest"]).resample(
                self.sampling_rate
            )
            # Select single channel if configured (for multi-channel audio files)
            channel = self.cfg.get("channel", None)
            if channel is not None:
                logging.info(f"Selecting channel {channel} for multi-channel audio")
                cutset = cutset.map(partial(_select_channel, channel=channel))
            hours = train_set["hours"]
            weight = train_set.get("weights", 1)
            lang = train_set.get("lang", "zh")
            if self.cfg.use_infinite_dataset:
                cutset = (
                    cutset.repeat()
                )  # this will result in infinite iterator that will never end
            else:
                cutset = cutset.repeat(
                    weight
                )  # each of the cutset will be repeated infinitely to make sure the iterator will not stop by them
            cutset[
                0
            ].load_audio()  # just to make sure we can get access to this cutset audio
            langs_hours[lang] += weight * hours
            cutset_hours.append(weight * hours)
            cutset_list.append(cutset)

        for lang in langs_hours:
            logging.info(
                f"Getting {langs_hours[lang]} hours of training data from {lang} language"
            )
        logging.info(
            f"Getting totally {sum(cutset_hours)} hours of training data from {len(cutset_hours)} manifests"
        )

        if len(cutset_list) > 1:  # more than 1 dataset
            logging.info("Muxing cuts")
            cutset_train = CutSet.mux(
                *cutset_list,
                weights=cutset_hours,
                stop_early=True,  # the epoch will stop when one of the dataset iterator is exhausted.
            )
        else:
            cutset_train = cutset_list[0]

        return cutset_train

    def setup_train(self):
        """Prepare the training DataLoader.

        Steps
        -----
        1) Load dataset specs from cfg.train_data_config (YAML).
        2) Build and filter the mixed training CutSet.
        3) Build a training sampler via BaseLhotseDatamodule utilities.
        4) Construct K2SpeechRecognitionDataset and the DataLoader.
        """
        with open(self.cfg.train_data_config, "r") as file:
            train_data_config = yaml.load(file, Loader=yaml.FullLoader)

        train_cutset = self._build_train_mux_cutset(train_data_config)
        train_cutset = self._filter_cutset(train_cutset, split="train")
        train_sampler = self._build_train_sampler(train_cutset)
        train_dataset = MultiSpeakerAsrDataset(
            input_strategy=self.input_strategy,
            cut_transforms=self.transforms,
            input_transforms=self.input_transforms,
            return_cuts=True,
        )
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)
        self.train_dl = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.cfg.get("num_workers", 8),
            persistent_workers=self.cfg.get("num_workers", 8) > 0,
            worker_init_fn=worker_init_fn,
        )

    def setup_valid(self):
        """Prepare validation DataLoaders (possibly multiple).

        The cfg.valid_data_config YAML contains a list of entries with:
        - manifest (str): absolute path to Lhotse CutSet jsonl.gz
        - name (str): a short tag used for logging/evaluation

        For each entry, we build a CutSet, apply the same filtering as training
        (without padding unless configured), and create a DataLoader using
        DynamicBucketingSampler with shuffle disabled.
        """
        with open(self.cfg.valid_data_config, "r") as file:
            valid_data_config = yaml.load(file, Loader=yaml.FullLoader)

        self.valid_dls = []
        self.valid_names = []
        for valid_set in valid_data_config:
            logging.info(f"Getting validation cuts: {valid_set['manifest']}")
            cutset = CutSet.from_file(valid_set["manifest"]).resample(
                self.sampling_rate
            )
            # Select single channel if configured (for multi-channel audio files)
            channel = self.cfg.get("channel", None)
            if channel is not None:
                logging.info(f"Selecting channel {channel} for multi-channel audio")
                cutset = cutset.map(partial(_select_channel, channel=channel))
            cutset = self._filter_cutset(cutset, split="valid")
            valid_name = valid_set["name"]

            valid_dataset = MultiSpeakerAsrDataset(
                input_strategy=self.input_strategy,
                return_cuts=True,
            )

            valid_sampler = DynamicBucketingSampler(
                cutset,
                max_duration=self.cfg.sampler.max_duration,
                shuffle=False,
            )
            # Use fewer workers for validation to avoid CPU OOM
            # (LLM takes lots of GPU memory, leaving less for CPU data loading)
            valid_num_workers = min(2, self.cfg.get("num_workers", 8))
            # If num_workers > 0, use persistent_workers to avoid hanging
            use_persistent = valid_num_workers > 0
            valid_dl = DataLoader(
                valid_dataset,
                sampler=valid_sampler,
                batch_size=None,
                num_workers=valid_num_workers,
                persistent_workers=use_persistent,
            )

            self.valid_names.append(valid_name)
            self.valid_dls.append(valid_dl)
