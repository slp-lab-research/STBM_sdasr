#!/usr/bin/env python3
"""
Script for multi-speaker Diarization and ASR evaluation using meeteval.
cpWER, globalWER, DER, Speaker Count metrics on xml-outputs.
# Gender Accuracy is optional
"""

import logging
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Set
import numpy as np
from scipy.optimize import linear_sum_assignment

from meeteval.wer.wer.cp import cp_word_error_rate
from meeteval.der import md_eval

from utils.xml_utils import parse_xml_to_segments, SpeakerSegment
from auden.utils.text_normalization import text_normalization


# md-eval invokes Perl subprocesses.  Cluster nodes do not consistently provide
# the locale inherited from the login node, so use Perl's universally available
# baseline locale for deterministic scoring without locale warnings.
os.environ["LANG"] = "C"
os.environ["LC_ALL"] = "C"
os.environ.pop("LANGUAGE", None)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def parse_xml_outputs_file(file_path: str) -> List[Tuple[str, str, str]]:
    """Parse xml-outputs file to extract (cut_id, gt_xml, hyp_xml) tuples.

    Args:
        file_path: Path to xml-outputs file

    Returns:
        List of (cut_id, gt_xml, hyp_xml) tuples
    """
    results = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by separator
    samples = content.split('=' * 80)

    for sample in samples:
        if not sample.strip():
            continue

        # Extract Cut ID
        cut_id_match = re.search(r'Cut ID:\s*(.+)', sample)
        if not cut_id_match:
            continue
        cut_id = cut_id_match.group(1).strip()

        # Extract Ground Truth XML
        gt_match = re.search(
            r'Ground Truth XML:\n(.+?)\n-{40}',
            sample,
            re.DOTALL
        )
        if not gt_match:
            continue
        gt_xml = gt_match.group(1).strip()

        # Extract Hypothesis XML (everything after "Hypothesis XML:" until end of sample)
        hyp_match = re.search(
            r'Hypothesis XML:\n(.+)',
            sample,
            re.DOTALL
        )
        if not hyp_match:
            continue
        hyp_xml = hyp_match.group(1).strip()

        results.append((cut_id, gt_xml, hyp_xml))

    return results


def parse_separate_xml_files(ref_file: str, hyp_file: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Parse separate reference and hypothesis XML files.

    Args:
        ref_file: Path to reference XML file
        hyp_file: Path to hypothesis XML file

    Returns:
        Tuple of (ref_dict, hyp_dict) where keys are cut_ids and values are XML strings
    """
    def parse_xml_file(file_path: str) -> Dict[str, str]:
        """Parse a single XML file with format:
        Cut ID: xxx
        <xml content>
        -------- (separator)
        """
        results = {}

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by separator (80 dashes)
        samples = content.split('-' * 80)

        for sample in samples:
            if not sample.strip():
                continue

            # Extract Cut ID
            cut_id_match = re.search(r'Cut ID:\s*(.+)', sample)
            if not cut_id_match:
                continue
            cut_id = cut_id_match.group(1).strip()

            # Extract XML content (everything after "Cut ID:" line)
            lines = sample.split('\n', 1)
            if len(lines) > 1:
                xml_content = lines[1].strip()
                if xml_content:
                    results[cut_id] = xml_content

        return results

    ref_dict = parse_xml_file(ref_file)
    hyp_dict = parse_xml_file(hyp_file)

    return ref_dict, hyp_dict


def normalize_text(text: str) -> str:
    """Normalize text using auden's text normalization."""
    normalized = text_normalization(
        text,
        case="lower",
        space_between_cjk=True,  # Keep True for character-level WER evaluation
        remove_diacritics=False,
        remove_symbols=True,
    )
    return normalized


def segments_to_cpwer_format(segments: List[SpeakerSegment]) -> List[str]:
    """Convert segments to cpWER format (list of strings, one per speaker).

    Args:
        segments: List of SpeakerSegment objects

    Returns:
        List of concatenated text per speaker, e.g., ['spk1_text', 'spk2_text']
    """
    # Group by speaker
    speaker_texts = defaultdict(list)
    for seg in segments:
        speaker_texts[seg.speaker_id].append(seg.text)

    # Sort by speaker_id and concatenate
    result = []
    for speaker_id in sorted(speaker_texts.keys()):
        # Concatenate all texts for this speaker
        full_text = ' '.join(speaker_texts[speaker_id])
        # Normalize
        normalized = normalize_text(full_text)
        result.append(normalized)

    return result


def segments_to_rttm_string(segments: List[SpeakerSegment], utt_id: str) -> str:
    """Convert segments to RTTM string format for DER calculation.

    Args:
        segments: List of SpeakerSegment objects
        utt_id: Utterance ID

    Returns:
        RTTM format string
    """
    # md-eval expects a speaker to be active at most once at any instant. Model
    # output can contain overlapping ranges for the same speaker, e.g.
    # 23.08-34.68 and 32.10-36.68. Their union is still one speaker turn, so
    # merge those ranges before producing RTTM. Overlap between different
    # speakers is intentionally retained and scored normally.
    intervals_by_speaker = defaultdict(list)
    for seg in segments:
        if seg.end_time > seg.start_time:
            intervals_by_speaker[seg.speaker_id].append(
                (seg.start_time, seg.end_time)
            )

    merged_segments = []
    for speaker_id, intervals in intervals_by_speaker.items():
        merged = []
        for start_time, end_time in sorted(intervals):
            if merged and start_time <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end_time))
            else:
                merged.append((start_time, end_time))
        merged_segments.extend(
            (start_time, end_time, speaker_id)
            for start_time, end_time in merged
        )

    lines = []
    for start_time, end_time, speaker_id in sorted(merged_segments):
        duration = end_time - start_time
        # RTTM format: SPEAKER <session> <channel> <start> <duration> <ortho> <stype> <speaker> <conf>
        line = f"SPEAKER {utt_id} 1 {start_time:.2f} {duration:.2f} <NA> <NA> spk{speaker_id} <NA>"
        lines.append(line)

    return '\n'.join(lines)


def calculate_speaker_count_metrics(gt_segments: List[SpeakerSegment],
                                    hyp_segments: List[SpeakerSegment]) -> Dict:
    """Calculate speaker count metrics.

    Returns:
        dict with exact_match (0 or 1) and absolute_error
    """
    # Count unique speakers
    gt_speakers = set(seg.speaker_id for seg in gt_segments)
    hyp_speakers = set(seg.speaker_id for seg in hyp_segments)

    num_gt = len(gt_speakers)
    num_hyp = len(hyp_speakers)

    exact_match = 1 if num_gt == num_hyp else 0
    absolute_error = abs(num_gt - num_hyp)

    return {
        'exact_match': exact_match,
        'absolute_error': absolute_error,
        'num_gt': num_gt,
        'num_hyp': num_hyp
    }


def calculate_gender_accuracy(gt_segments: List[SpeakerSegment],
                              hyp_segments: List[SpeakerSegment],
                              overlap_threshold: float = 0.75) -> Dict:
    """Calculate gender accuracy based on time overlap (OLD METHOD).

    For each GT speaker:
    1. Calculate total duration of GT speaker
    2. Calculate overlap time with each predicted speaker
    3. Match to the predicted speaker with maximum overlap
    4. Only consider valid if overlap >= threshold * GT_duration
    5. Check if gender matches

    Args:
        gt_segments: Ground truth segments
        hyp_segments: Hypothesis segments
        overlap_threshold: Minimum overlap ratio (default 0.75 = 75%)

    Returns:
        dict with correct, total, and accuracy
    """
    from collections import defaultdict

    # Group segments by speaker
    gt_by_speaker = defaultdict(list)
    for seg in gt_segments:
        gt_by_speaker[seg.speaker_id].append(seg)

    hyp_by_speaker = defaultdict(list)
    for seg in hyp_segments:
        hyp_by_speaker[seg.speaker_id].append(seg)

    if not gt_by_speaker or not hyp_by_speaker:
        return {'correct': 0, 'total': len(gt_by_speaker), 'accuracy': 0.0}

    correct = 0
    total = len(gt_by_speaker)

    # For each GT speaker
    for gt_spk_id, gt_segs in gt_by_speaker.items():
        # Calculate total duration of GT speaker
        gt_total_duration = sum(seg.end_time - seg.start_time for seg in gt_segs)

        # Calculate overlap with each hyp speaker
        overlaps = {}

        for hyp_spk_id, hyp_segs in hyp_by_speaker.items():
            total_overlap = 0.0

            # Calculate overlap between all segment pairs
            for gt_seg in gt_segs:
                for hyp_seg in hyp_segs:
                    overlap = max(0, min(gt_seg.end_time, hyp_seg.end_time) -
                                     max(gt_seg.start_time, hyp_seg.start_time))
                    total_overlap += overlap

            overlaps[hyp_spk_id] = total_overlap

        # Find hyp speaker with max overlap
        if overlaps:
            best_hyp_spk_id = max(overlaps, key=overlaps.get)
            max_overlap = overlaps[best_hyp_spk_id]

            # Check if overlap is sufficient (>= threshold * GT duration)
            if max_overlap >= overlap_threshold * gt_total_duration:
                # Get genders (normalize to lowercase)
                gt_gender = gt_segs[0].gender.lower() if gt_segs[0].gender else 'unknown'
                hyp_gender = hyp_by_speaker[best_hyp_spk_id][0].gender.lower() if hyp_by_speaker[best_hyp_spk_id][0].gender else 'unknown'

                # Map short forms
                gender_map = {'m': 'male', 'f': 'female', 'u': 'unknown'}
                gt_gender = gender_map.get(gt_gender, gt_gender)
                hyp_gender = gender_map.get(hyp_gender, hyp_gender)

                # Check if match
                if gt_gender == hyp_gender:
                    correct += 1
            # else: overlap too small, count as incorrect (don't increment correct)

    accuracy = correct / total if total > 0 else 0.0

    return {
        'correct': correct,
        'total': total,
        'accuracy': accuracy
    }


def calculate_overlap_time(segments_a: List[SpeakerSegment],
                           segments_b: List[SpeakerSegment]) -> float:
    """Calculate total overlap time between two lists of segments.

    Args:
        segments_a: First list of segments
        segments_b: Second list of segments

    Returns:
        Total overlap time in seconds
    """
    total_overlap = 0.0

    for seg_a in segments_a:
        for seg_b in segments_b:
            overlap = max(0, min(seg_a.end_time, seg_b.end_time) -
                             max(seg_a.start_time, seg_b.start_time))
            total_overlap += overlap

    return total_overlap


def normalize_gender(gender: str) -> str:
    """Normalize gender string to standard format.

    Args:
        gender: Gender string (can be 'm', 'f', 'male', 'female', etc.)

    Returns:
        Normalized gender ('male', 'female', or 'unknown')
    """
    if not gender:
        return 'unknown'

    gender = gender.lower().strip()
    gender_map = {
        'm': 'male',
        'f': 'female',
        'u': 'unknown',
        'male': 'male',
        'female': 'female',
        'unknown': 'unknown'
    }

    return gender_map.get(gender, 'unknown')


def get_optimal_speaker_mapping(gt_segments: List[SpeakerSegment],
                                hyp_segments: List[SpeakerSegment]) -> Dict[str, str]:
    """Get optimal speaker mapping based on maximum overlap (Hungarian algorithm).

    This implements the same alignment strategy as DER calculation.

    Args:
        gt_segments: Ground truth segments
        hyp_segments: Hypothesis segments

    Returns:
        Dictionary mapping {hyp_speaker_id: gt_speaker_id}
    """
    from collections import defaultdict

    # Group segments by speaker
    gt_by_spk = defaultdict(list)
    for seg in gt_segments:
        gt_by_spk[seg.speaker_id].append(seg)

    hyp_by_spk = defaultdict(list)
    for seg in hyp_segments:
        hyp_by_spk[seg.speaker_id].append(seg)

    if not gt_by_spk or not hyp_by_spk:
        return {}

    # Get sorted speaker lists for consistent indexing
    gt_spks = sorted(gt_by_spk.keys())
    hyp_spks = sorted(hyp_by_spk.keys())

    # Build overlap matrix: overlap_matrix[i][j] = overlap time between gt_spks[i] and hyp_spks[j]
    overlap_matrix = np.zeros((len(gt_spks), len(hyp_spks)))

    for i, gt_spk in enumerate(gt_spks):
        for j, hyp_spk in enumerate(hyp_spks):
            overlap_matrix[i, j] = calculate_overlap_time(
                gt_by_spk[gt_spk],
                hyp_by_spk[hyp_spk]
            )

    # Apply Hungarian algorithm to maximize overlap (minimize -overlap)
    row_ind, col_ind = linear_sum_assignment(-overlap_matrix)

    # Build mapping: only include pairs with non-zero overlap
    mapping = {}
    for i, j in zip(row_ind, col_ind):
        if overlap_matrix[i, j] > 0:
            mapping[hyp_spks[j]] = gt_spks[i]

    return mapping


def calculate_gender_accuracy_der_based(gt_segments: List[SpeakerSegment],
                                        hyp_segments: List[SpeakerSegment],
                                        include_fa_miss: bool = False) -> Dict:
    """Calculate gender accuracy based on DER-style alignment (NEW METHOD).

    Uses optimal speaker mapping (Hungarian algorithm) to align GT and Hyp speakers
    based on maximum time overlap, then checks gender accuracy for aligned pairs.

    Algorithm:
    1. Get optimal speaker mapping using Hungarian algorithm (same as DER)
    2. For each aligned speaker pair, calculate their overlap time
    3. Check if genders match
    4. Gender Accuracy = (sum of correct overlap time) / (sum of total overlap time)

    Args:
        gt_segments: Ground truth segments
        hyp_segments: Hypothesis segments
        include_fa_miss: Whether to include FA/Miss in denominator (default: False)

    Returns:
        dict with accuracy, correct_time, total_time, and mapping details
    """
    from collections import defaultdict

    # Group segments by speaker
    gt_by_spk = defaultdict(list)
    for seg in gt_segments:
        gt_by_spk[seg.speaker_id].append(seg)

    hyp_by_spk = defaultdict(list)
    for seg in hyp_segments:
        hyp_by_spk[seg.speaker_id].append(seg)

    if not gt_by_spk or not hyp_by_spk:
        return {
            'accuracy': 0.0,
            'correct_time': 0.0,
            'total_time': 0.0,
            'num_aligned_pairs': 0,
            'num_correct_pairs': 0,
            'mapping': {}
        }

    # Step 1: Get optimal speaker mapping (DER-based alignment)
    mapping = get_optimal_speaker_mapping(gt_segments, hyp_segments)

    # Step 2: Calculate gender accuracy for aligned pairs
    total_time = 0.0
    correct_time = 0.0
    num_correct_pairs = 0

    for hyp_spk, gt_spk in mapping.items():
        # Calculate overlap time for this aligned pair
        overlap_time = calculate_overlap_time(
            gt_by_spk[gt_spk],
            hyp_by_spk[hyp_spk]
        )

        if overlap_time > 0:
            total_time += overlap_time

            # Get genders
            gt_gender = normalize_gender(gt_by_spk[gt_spk][0].gender)
            hyp_gender = normalize_gender(hyp_by_spk[hyp_spk][0].gender)

            # Check if genders match
            if gt_gender == hyp_gender:
                correct_time += overlap_time
                num_correct_pairs += 1

    # Optional: Include FA and Miss in denominator
    if include_fa_miss:
        # FA: hyp speakers not in mapping (false alarm speakers)
        fa_speakers = set(hyp_by_spk.keys()) - set(mapping.keys())
        for hyp_spk in fa_speakers:
            fa_time = sum(seg.end_time - seg.start_time for seg in hyp_by_spk[hyp_spk])
            total_time += fa_time
            # FA contributes to total but not to correct (gender is "wrong" by definition)

        # Miss: gt speakers not in mapping values (missed speakers)
        mapped_gt = set(mapping.values())
        miss_speakers = set(gt_by_spk.keys()) - mapped_gt
        for gt_spk in miss_speakers:
            miss_time = sum(seg.end_time - seg.start_time for seg in gt_by_spk[gt_spk])
            total_time += miss_time
            # Miss contributes to total but not to correct

    accuracy = correct_time / total_time if total_time > 0 else 0.0

    return {
        'accuracy': accuracy,
        'correct_time': correct_time,
        'total_time': total_time,
        'num_aligned_pairs': len(mapping),
        'num_correct_pairs': num_correct_pairs,
        'mapping': mapping
    }


def evaluate_single_sample(cut_id: str, gt_xml: str, hyp_xml: str,
                          collar_values: List[float] = [0.25],
                          verbose: bool = True):
    """Evaluate a single sample for cpWER, DER, Gender Accuracy, and Speaker Count.

    Args:
        cut_id: Sample ID
        gt_xml: Ground truth XML
        hyp_xml: Hypothesis XML
        collar_values: List of collar values for DER calculation
        verbose: Whether to print detailed results

    Returns:
        Dict with cpwer, der, gender (DER-based), and count results, or None if parsing failed
    """
    try:
        # Parse XML
        gt_segments = parse_xml_to_segments(gt_xml)
        hyp_segments = parse_xml_to_segments(hyp_xml)

        if not gt_segments:
            logging.warning(f"[{cut_id}] Ground truth XML parsing failed")
            return None
        if not hyp_segments:
            logging.warning(f"[{cut_id}] Hypothesis XML parsing failed")
            return None

        if verbose:
            print(f"\n{'='*80}")
            print(f"Sample: {cut_id}")
            print(f"{'='*80}")

            print(f"\n--- Parsed GT Segments ({len(gt_segments)} segments) ---")
            for seg in gt_segments[:3]:  # Show first 3
                print(f"  Speaker {seg.speaker_id}: [{seg.start_time:.2f}-{seg.end_time:.2f}] {seg.text[:50]}...")
            if len(gt_segments) > 3:
                print(f"  ... and {len(gt_segments) - 3} more segments")

            print(f"\n--- Parsed Hyp Segments ({len(hyp_segments)} segments) ---")
            for seg in hyp_segments[:3]:  # Show first 3
                print(f"  Speaker {seg.speaker_id}: [{seg.start_time:.2f}-{seg.end_time:.2f}] {seg.text[:50]}...")
            if len(hyp_segments) > 3:
                print(f"  ... and {len(hyp_segments) - 3} more segments")

        # 1. Calculate cpWER
        gt_texts = segments_to_cpwer_format(gt_segments)
        hyp_texts = segments_to_cpwer_format(hyp_segments)

        if verbose:
            print(f"\n--- cpWER Input ---")
            print(f"GT speakers: {len(gt_texts)}")
            for i, text in enumerate(gt_texts, 1):
                print(f"  Speaker {i}: {text[:100]}...")
            print(f"Hyp speakers: {len(hyp_texts)}")
            for i, text in enumerate(hyp_texts, 1):
                print(f"  Speaker {i}: {text[:100]}...")

        cpwer_result = cp_word_error_rate(reference=gt_texts, hypothesis=hyp_texts)

        if verbose:
            print(f"\n--- cpWER Result ---")
            print(f"  Error Rate: {cpwer_result.error_rate:.2%}")
            print(f"  Errors: {cpwer_result.errors}")
            print(f"  Length: {cpwer_result.length}")
            print(f"  Insertions: {cpwer_result.insertions}")
            print(f"  Deletions: {cpwer_result.deletions}")
            print(f"  Substitutions: {cpwer_result.substitutions}")
            print(f"  Assignment: {cpwer_result.assignment}")

        # 1b. Calculate Global WER (concatenate all text by time, ignore speakers)
        # Sort by start time
        gt_sorted = sorted(gt_segments, key=lambda x: x.start_time)
        hyp_sorted = sorted(hyp_segments, key=lambda x: x.start_time)

        # Concatenate all text and split into words
        gt_words = []
        for seg in gt_sorted:
            normalized = normalize_text(seg.text)
            gt_words.extend(normalized.split())

        hyp_words = []
        for seg in hyp_sorted:
            normalized = normalize_text(seg.text)
            hyp_words.extend(normalized.split())

        # Calculate WER using kaldialign
        import kaldialign
        ERR = "*"
        ali = kaldialign.align(gt_words, hyp_words, ERR)

        # Count errors
        num_ins = 0
        num_del = 0
        num_sub = 0
        num_corr = 0

        for ref_word, hyp_word in ali:
            if ref_word == ERR:
                num_ins += 1
            elif hyp_word == ERR:
                num_del += 1
            elif ref_word != hyp_word:
                num_sub += 1
            else:
                num_corr += 1

        ref_len = len(gt_words)
        total_errors = num_ins + num_del + num_sub
        error_rate = total_errors / ref_len if ref_len > 0 else 0.0

        # Create result object similar to meeteval
        class GlobalWERResult:
            def __init__(self, errors, length, insertions, deletions, substitutions):
                self.errors = errors
                self.length = length
                self.insertions = insertions
                self.deletions = deletions
                self.substitutions = substitutions
                self.error_rate = errors / length if length > 0 else 0.0

        global_wer_result = GlobalWERResult(
            errors=total_errors,
            length=ref_len,
            insertions=num_ins,
            deletions=num_del,
            substitutions=num_sub
        )

        if verbose:
            print(f"\n--- Global WER (speaker-agnostic) ---")
            print(f"  Error Rate: {global_wer_result.error_rate:.2%}")
            print(f"  Errors: {global_wer_result.errors}")
            print(f"  Length: {global_wer_result.length}")
            print(f"  Insertions: {global_wer_result.insertions}")
            print(f"  Deletions: {global_wer_result.deletions}")
            print(f"  Substitutions: {global_wer_result.substitutions}")

        # 2. Calculate DER with multiple collar values
        from meeteval.io import RTTM

        gt_rttm_str = segments_to_rttm_string(gt_segments, cut_id)
        hyp_rttm_str = segments_to_rttm_string(hyp_segments, cut_id)

        gt_rttm = RTTM.parse(gt_rttm_str)
        hyp_rttm = RTTM.parse(hyp_rttm_str)

        if verbose:
            print(f"\n--- DER Input ---")
            print(f"GT RTTM entries: {len(gt_rttm)}")
            print("GT RTTM string (first 3 lines):")
            for line in gt_rttm_str.strip().split('\n')[:3]:
                print(f"  {line}")
            print(f"Hyp RTTM entries: {len(hyp_rttm)}")
            print("Hyp RTTM string (first 3 lines):")
            for line in hyp_rttm_str.strip().split('\n')[:3]:
                print(f"  {line}")

        # Calculate DER for each collar value
        der_results = {}
        der_failed = False
        for collar in collar_values:
            try:
                der_result = md_eval.md_eval_22(
                    reference=gt_rttm,
                    hypothesis=hyp_rttm,
                    collar=collar,
                )
                der_results[collar] = der_result

                if verbose:
                    print(f"\n--- DER Result (collar={collar}) ---")
                    print(f"  DER: {float(der_result.error_rate):.2%}")
                    print(f"  Scored Speaker Time: {float(der_result.scored_speaker_time):.2f}s")
                    print(f"  Missed Speaker Time: {float(der_result.missed_speaker_time):.2f}s")
                    print(f"  False Alarm Time: {float(der_result.falarm_speaker_time):.2f}s")
                    print(f"  Speaker Error Time: {float(der_result.speaker_error_time):.2f}s")
            except Exception as e:
                der_failed = True
                logging.warning(f"[{cut_id}] DER calculation failed (collar={collar}): {e}")
                if verbose:
                    print(f"\n--- DER Result (collar={collar}) ---")
                    print(f"  DER calculation failed: {e}")

        # 3. Calculate Gender Accuracy - DER-based alignment (time-weighted)
        gender_result = calculate_gender_accuracy_der_based(gt_segments, hyp_segments, include_fa_miss=False)

        if verbose:
            print(f"\n--- Gender Accuracy (DER-based, time-weighted) ---")
            print(f"  Accuracy: {gender_result['accuracy']:.2%}")
            print(f"  Correct Time: {gender_result['correct_time']:.2f}s")
            print(f"  Total Time: {gender_result['total_time']:.2f}s")
            print(f"  Aligned Pairs: {gender_result['num_aligned_pairs']}")
            print(f"  Correct Pairs: {gender_result['num_correct_pairs']}/{gender_result['num_aligned_pairs']}")
            print(f"  Speaker Mapping: {gender_result['mapping']}")

        # 4. Calculate Speaker Count Metrics
        count_result = calculate_speaker_count_metrics(gt_segments, hyp_segments)

        if verbose:
            print(f"\n--- Speaker Count ---")
            print(f"  GT Speakers: {count_result['num_gt']}")
            print(f"  Hyp Speakers: {count_result['num_hyp']}")
            print(f"  Exact Match: {'Yes' if count_result['exact_match'] else 'No'}")
            print(f"  Absolute Error: {count_result['absolute_error']}")

        return {
            'cpwer': cpwer_result,
            'global_wer': global_wer_result,  # Speaker-agnostic WER
            'der': der_results,  # Dict with collar as key (may be empty if DER failed)
            'der_failed': der_failed,
            'gender': gender_result,  # DER-based, time-weighted
            'count': count_result,
            'gt_segments': gt_segments,
            'hyp_segments': hyp_segments,
        }

    except Exception as e:
        logging.error(f"[{cut_id}] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_full_dataset(xml_outputs_file: str = None,
                         collar_values: List[float] = [0.25, 0.0],
                         ref_file: str = None, hyp_file: str = None,
                         output_failed_samples: str = None,
                         filter_cut_ids_file: str = None):
    """Evaluate the full dataset and compute aggregate metrics.

    Args:
        xml_outputs_file: Path to xml-outputs file (old format, gt and hyp together)
        collar_values: List of collar values for DER calculation
        ref_file: Path to reference XML file (new format)
        hyp_file: Path to hypothesis XML file (new format)
        output_failed_samples: Path to output file for failed sample IDs
        filter_cut_ids_file: Path to text file containing cut IDs to evaluate (one per line)
    """
    print(f"\n{'='*80}")
    if ref_file and hyp_file:
        print(f"Evaluating Full Dataset:")
        print(f"  Reference: {ref_file}")
        print(f"  Hypothesis: {hyp_file}")
    else:
        print(f"Evaluating Full Dataset: {xml_outputs_file}")
    print(f"DER Collars: {collar_values}")
    if filter_cut_ids_file:
        print(f"Filter Cut IDs: {filter_cut_ids_file}")
    print(f"{'='*80}\n")

    # Load filter cut IDs if provided
    filter_cut_ids = None
    if filter_cut_ids_file:
        print(f"Loading filter cut IDs from: {filter_cut_ids_file}")
        with open(filter_cut_ids_file, 'r', encoding='utf-8') as f:
            filter_cut_ids = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(filter_cut_ids)} cut IDs to filter\n")

    # Parse files
    total_ref_samples = 0
    if ref_file and hyp_file:
        print("Parsing separate reference and hypothesis XML files...")
        ref_dict, hyp_dict = parse_separate_xml_files(ref_file, hyp_file)
        print(f"Found {len(ref_dict)} reference samples")
        print(f"Found {len(hyp_dict)} hypothesis samples")

        # Apply filter if provided
        if filter_cut_ids:
            print(f"\nApplying cut ID filter...")
            ref_dict = {k: v for k, v in ref_dict.items() if k in filter_cut_ids}
            hyp_dict = {k: v for k, v in hyp_dict.items() if k in filter_cut_ids}
            print(f"After filtering: {len(ref_dict)} reference samples, {len(hyp_dict)} hypothesis samples")

        # Find common cut IDs
        ref_ids = set(ref_dict.keys())
        hyp_ids = set(hyp_dict.keys())
        common_ids = ref_ids & hyp_ids
        missing_in_hyp = ref_ids - hyp_ids
        extra_in_hyp = hyp_ids - ref_ids

        total_ref_samples = len(ref_ids)  # Store total reference samples

        print(f"\n{'='*80}")
        print(f"CUT ID STATISTICS")
        print(f"{'='*80}")
        print(f"Total Reference Cuts: {len(ref_ids)}")
        print(f"Total Hypothesis Cuts: {len(hyp_ids)}")
        print(f"Common Cuts (for evaluation): {len(common_ids)}")
        print(f"Missing in Hypothesis: {len(missing_in_hyp)} ({len(missing_in_hyp)/len(ref_ids)*100:.2f}%)")
        if missing_in_hyp and len(missing_in_hyp) <= 20:
            print(f"  Missing Cut IDs: {sorted(list(missing_in_hyp))}")
        elif missing_in_hyp:
            print(f"  Sample Missing Cut IDs (first 20): {sorted(list(missing_in_hyp))[:20]}")
        print(f"Extra in Hypothesis: {len(extra_in_hyp)} ({len(extra_in_hyp)/len(ref_ids)*100:.2f}%)")
        if extra_in_hyp and len(extra_in_hyp) <= 20:
            print(f"  Extra Cut IDs: {sorted(list(extra_in_hyp))}")
        elif extra_in_hyp:
            print(f"  Sample Extra Cut IDs (first 20): {sorted(list(extra_in_hyp))[:20]}")
        print(f"{'='*80}\n")

        # Create samples list from common IDs
        samples = [(cut_id, ref_dict[cut_id], hyp_dict[cut_id]) for cut_id in sorted(common_ids)]
    else:
        print("Parsing xml-outputs file...")
        samples = parse_xml_outputs_file(xml_outputs_file)
        print(f"Found {len(samples)} samples")

        # Apply filter if provided
        if filter_cut_ids:
            print(f"Applying cut ID filter...")
            samples = [(cut_id, gt_xml, hyp_xml) for cut_id, gt_xml, hyp_xml in samples if cut_id in filter_cut_ids]
            print(f"After filtering: {len(samples)} samples")
        print()

    # Aggregate statistics
    total_cpwer_errors = 0
    total_cpwer_length = 0
    total_cpwer_insertions = 0
    total_cpwer_deletions = 0
    total_cpwer_substitutions = 0

    # Global WER statistics
    total_global_wer_errors = 0
    total_global_wer_length = 0
    total_global_wer_insertions = 0
    total_global_wer_deletions = 0
    total_global_wer_substitutions = 0

    # Initialize from the first valid result. Some meeteval versions implement
    # DiaErrorRate.zero() as 0/0 and raise ZeroDivisionError.
    total_der = {collar: None for collar in collar_values}

    # Gender Accuracy: DER-based (time-weighted)
    total_gender_correct_time = 0.0
    total_gender_total_time = 0.0
    total_gender_correct_pairs = 0
    total_gender_total_pairs = 0

    # Speaker Count
    total_count_exact_match = 0
    total_count_mae = 0.0

    failed_samples = []
    parse_failed_ref = []
    parse_failed_hyp = []
    parse_failed_both = []
    der_failed_samples = []
    successful_samples = 0

    # Process all samples
    for idx, (cut_id, gt_xml, hyp_xml) in enumerate(samples):
        if (idx + 1) % 100 == 0:
            print(f"Processing {idx + 1}/{len(samples)}...")

        result = evaluate_single_sample(cut_id, gt_xml, hyp_xml,
                                       collar_values=collar_values,
                                       verbose=False)

        if result is None:
            # Try to determine which parsing failed
            try:
                gt_segments = parse_xml_to_segments(gt_xml)
                gt_parsed = bool(gt_segments)
            except:
                gt_parsed = False

            try:
                hyp_segments = parse_xml_to_segments(hyp_xml)
                hyp_parsed = bool(hyp_segments)
            except:
                hyp_parsed = False

            if not gt_parsed and not hyp_parsed:
                parse_failed_both.append(cut_id)
            elif not gt_parsed:
                parse_failed_ref.append(cut_id)
            elif not hyp_parsed:
                parse_failed_hyp.append(cut_id)

            failed_samples.append(cut_id)
            continue

        # Accumulate cpWER
        total_cpwer_errors += result['cpwer'].errors
        total_cpwer_length += result['cpwer'].length
        total_cpwer_insertions += result['cpwer'].insertions
        total_cpwer_deletions += result['cpwer'].deletions
        total_cpwer_substitutions += result['cpwer'].substitutions

        # Accumulate Global WER
        total_global_wer_errors += result['global_wer'].errors
        total_global_wer_length += result['global_wer'].length
        total_global_wer_insertions += result['global_wer'].insertions
        total_global_wer_deletions += result['global_wer'].deletions
        total_global_wer_substitutions += result['global_wer'].substitutions

        # Accumulate DER for each collar value (DiaErrorRate objects can be added)
        # Only accumulate if DER didn't fail for this sample
        if result.get('der_failed', False):
            der_failed_samples.append(cut_id)
        else:
            for collar in collar_values:
                if collar in result['der']:
                    if total_der[collar] is None:
                        total_der[collar] = result['der'][collar]
                    else:
                        total_der[collar] = (
                            total_der[collar] + result['der'][collar]
                        )

        # Accumulate Gender Accuracy (DER-based, time-weighted)
        total_gender_correct_time += result['gender']['correct_time']
        total_gender_total_time += result['gender']['total_time']
        total_gender_correct_pairs += result['gender']['num_correct_pairs']
        total_gender_total_pairs += result['gender']['num_aligned_pairs']

        # Accumulate Speaker Count
        total_count_exact_match += result['count']['exact_match']
        total_count_mae += result['count']['absolute_error']

        successful_samples += 1

    # Print aggregate results
    print(f"\n{'='*80}")
    print(f"AGGREGATE RESULTS")
    print(f"{'='*80}\n")

    print(f"--- Processing Statistics ---")
    if total_ref_samples > 0:
        print(f"Total Reference Samples: {total_ref_samples}")
        print(f"Samples available for evaluation: {len(samples)} ({len(samples)/total_ref_samples*100:.2f}%)")
        print(f"Successfully evaluated: {successful_samples} ({successful_samples/total_ref_samples*100:.2f}%)")
        print(f"Parse failed: {len(failed_samples)} ({len(failed_samples)/total_ref_samples*100:.2f}%)")
        print(f"Total failed: {total_ref_samples-successful_samples} ({(total_ref_samples-successful_samples)/total_ref_samples*100:.2f}%)")
    else:
        print(f"Total samples for evaluation: {len(samples)}")
        print(f"Successfully evaluated: {successful_samples}")
        print(f"Parse failed: {len(failed_samples)} ({len(failed_samples)/len(samples)*100:.2f}%)")

    if parse_failed_ref:
        print(f"\n  Reference XML parse failed: {len(parse_failed_ref)}")
        if len(parse_failed_ref) <= 10:
            print(f"    Cut IDs: {parse_failed_ref}")
        else:
            print(f"    Sample Cut IDs (first 10): {parse_failed_ref[:10]}")
            print(f"    ... and {len(parse_failed_ref) - 10} more")

    if parse_failed_hyp:
        print(f"\n  Hypothesis XML parse failed: {len(parse_failed_hyp)}")
        if len(parse_failed_hyp) <= 10:
            print(f"    Cut IDs: {parse_failed_hyp}")
        else:
            print(f"    Sample Cut IDs (first 10): {parse_failed_hyp[:10]}")
            print(f"    ... and {len(parse_failed_hyp) - 10} more")

    if parse_failed_both:
        print(f"\n  Both Reference and Hypothesis parse failed: {len(parse_failed_both)}")
        if len(parse_failed_both) <= 10:
            print(f"    Cut IDs: {parse_failed_both}")
        else:
            print(f"    Sample Cut IDs (first 10): {parse_failed_both[:10]}")
            print(f"    ... and {len(parse_failed_both) - 10} more")

    if der_failed_samples:
        print(f"\n  DER calculation failed (division by zero): {len(der_failed_samples)}")
        print(f"    (These samples were still evaluated for cpWER, Gender, and Count)")
        if len(der_failed_samples) <= 10:
            print(f"    Cut IDs: {der_failed_samples}")
        else:
            print(f"    Sample Cut IDs (first 10): {der_failed_samples[:10]}")
            print(f"    ... and {len(der_failed_samples) - 10} more")

    print()

    print(f"\n--- cpWER (Concatenated Minimum Permutation WER) ---")
    overall_cpwer = total_cpwer_errors / total_cpwer_length if total_cpwer_length > 0 else 0
    print(f"  cpWER: {overall_cpwer:.2%}")
    print(f"  Total Errors: {total_cpwer_errors}")
    print(f"  Total Length: {total_cpwer_length}")
    print(f"  Insertions: {total_cpwer_insertions}")
    print(f"  Deletions: {total_cpwer_deletions}")
    print(f"  Substitutions: {total_cpwer_substitutions}")

    print(f"\n--- Global WER (Speaker-Agnostic, Pure ASR) ---")
    overall_global_wer = total_global_wer_errors / total_global_wer_length if total_global_wer_length > 0 else 0
    print(f"  Global WER: {overall_global_wer:.2%}")
    print(f"  Total Errors: {total_global_wer_errors}")
    print(f"  Total Length: {total_global_wer_length}")
    print(f"  Insertions: {total_global_wer_insertions}")
    print(f"  Deletions: {total_global_wer_deletions}")
    print(f"  Substitutions: {total_global_wer_substitutions}")

    # DER results for each collar value
    print(f"\n--- DER (Diarization Error Rate) ---")
    if der_failed_samples:
        print(f"  Note: {len(der_failed_samples)} samples failed DER calculation (excluded from DER statistics)")

    for collar in collar_values:
        if successful_samples > 0 and total_der[collar] is not None:
            der = total_der[collar]
            scored_time = float(der.scored_speaker_time)

            if scored_time > 0:
                # Calculate percentages relative to scored time
                miss_pct = float(der.missed_speaker_time) / scored_time * 100 if scored_time > 0 else 0
                fa_pct = float(der.falarm_speaker_time) / scored_time * 100 if scored_time > 0 else 0
                conf_pct = float(der.speaker_error_time) / scored_time * 100 if scored_time > 0 else 0

                der_success_count = successful_samples - len(der_failed_samples)
                print(f"\n  Collar={collar}s (evaluated on {der_success_count} samples):")
                print(f"    Overall DER: {float(der.error_rate):.2%}")
                print(f"    Miss%: {miss_pct:.2f}%")
                print(f"    FA%: {fa_pct:.2f}%")
                print(f"    Conf%: {conf_pct:.2f}%")
                print(f"    Total Scored Speaker Time: {scored_time:.2f}s")
            else:
                print(f"\n  Collar={collar}s: No valid DER results")
        else:
            print(f"\n  Collar={collar}s: No DER results available")

    # Gender Accuracy results (DER-based, time-weighted)
    print(f"\n--- Gender Accuracy (DER-based, time-weighted) ---")
    if successful_samples > 0 and total_gender_total_time > 0:
        gender_acc = total_gender_correct_time / total_gender_total_time
        print(f"  Overall Gender Accuracy (time-weighted): {gender_acc:.2%}")
        print(f"  Correct Time: {total_gender_correct_time:.2f}s / {total_gender_total_time:.2f}s")
        print(f"  Correct Pairs: {total_gender_correct_pairs}/{total_gender_total_pairs} aligned speaker pairs")
        if total_gender_total_pairs > 0:
            pair_acc = total_gender_correct_pairs / total_gender_total_pairs
            print(f"  Pair-level Accuracy (unweighted): {pair_acc:.2%}")
    else:
        print("  No gender results available")

    print(f"\n--- Speaker Count Metrics ---")
    if successful_samples > 0:
        count_exact_match_rate = total_count_exact_match / successful_samples
        count_mae = total_count_mae / successful_samples
        print(f"  Exact Match Rate: {count_exact_match_rate:.2%} ({total_count_exact_match}/{successful_samples} samples)")
        print(f"  MAE (Mean Absolute Error): {count_mae:.2f}")
    else:
        print("  No count results available")

    print(f"\n{'='*80}\n")

    # Save failed samples to file if requested
    if output_failed_samples and (parse_failed_ref or parse_failed_hyp or parse_failed_both or der_failed_samples):
        with open(output_failed_samples, 'w', encoding='utf-8') as f:
            f.write("Failed Samples Analysis\n")
            f.write("="*80 + "\n\n")

            if parse_failed_ref:
                f.write(f"Reference XML Parse Failed ({len(parse_failed_ref)} samples):\n")
                f.write("-"*80 + "\n")
                for cut_id in sorted(parse_failed_ref):
                    f.write(f"{cut_id}\n")
                f.write("\n\n")

            if parse_failed_hyp:
                f.write(f"Hypothesis XML Parse Failed ({len(parse_failed_hyp)} samples):\n")
                f.write("-"*80 + "\n")
                for cut_id in sorted(parse_failed_hyp):
                    f.write(f"{cut_id}\n")
                f.write("\n\n")

            if parse_failed_both:
                f.write(f"Both Reference and Hypothesis Parse Failed ({len(parse_failed_both)} samples):\n")
                f.write("-"*80 + "\n")
                for cut_id in sorted(parse_failed_both):
                    f.write(f"{cut_id}\n")
                f.write("\n\n")

            if der_failed_samples:
                f.write(f"DER Calculation Failed - Division by Zero ({len(der_failed_samples)} samples):\n")
                f.write("(These samples were still evaluated for cpWER, Gender Accuracy, and Speaker Count)\n")
                f.write("-"*80 + "\n")
                for cut_id in sorted(der_failed_samples):
                    f.write(f"{cut_id}\n")
                f.write("\n\n")

        print(f"Failed samples list saved to: {output_failed_samples}")


def main():
    """Main test function."""
    import sys
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate multi-speaker ASR with cpWER, DER, Speaker Count Acc (and Gender Acc)')
    parser.add_argument('--xml_file', type=str, default=None,
                       help='Path to xml-outputs file (old format with gt and hyp together)')
    parser.add_argument('--ref_file', type=str, default=None,
                       help='Path to reference XML file (new format)')
    parser.add_argument('--hyp_file', type=str, default=None,
                       help='Path to hypothesis XML file (new format)')
    parser.add_argument('--test_first', action='store_true',
                       help='Test on first sample before full evaluation')
    parser.add_argument('--output_failed', type=str, default=None,
                       help='Output file path for failed samples list')
    parser.add_argument('--filter_cut_ids', type=str, default=None,
                       help='Path to text file containing cut IDs to evaluate (one per line)')

    args = parser.parse_args()

    # Validate arguments
    if not args.xml_file and not (args.ref_file and args.hyp_file):
        parser.error("Either --xml_file or both --ref_file and --hyp_file must be provided")

    if args.xml_file and (args.ref_file or args.hyp_file):
        parser.error("Cannot use --xml_file together with --ref_file/--hyp_file")

    if args.ref_file and args.hyp_file:
        print(f"Reference file: {args.ref_file}")
        print(f"Hypothesis file: {args.hyp_file}")
    else:
        print(f"XML file: {args.xml_file}")

    # Optional: Test on first sample
    if args.test_first:
        print("="*80)
        print("STEP 1: Testing on First Sample")
        print("="*80)

        if args.ref_file and args.hyp_file:
            ref_dict, hyp_dict = parse_separate_xml_files(args.ref_file, args.hyp_file)
            common_ids = set(ref_dict.keys()) & set(hyp_dict.keys())
            if not common_ids:
                print("No common samples found!")
                return
            cut_id = sorted(common_ids)[0]
            gt_xml = ref_dict[cut_id]
            hyp_xml = hyp_dict[cut_id]
        else:
            samples = parse_xml_outputs_file(args.xml_file)
            if not samples:
                print("No samples found!")
                return
            cut_id, gt_xml, hyp_xml = samples[0]

        print(f"\nTesting sample: {cut_id}")
        print(f"\n--- Raw Ground Truth XML ---")
        print(gt_xml[:500])
        print(f"\n--- Raw Hypothesis XML ---")
        print(hyp_xml[:500])

        result = evaluate_single_sample(cut_id, gt_xml, hyp_xml,
                                       collar_values=[0.25, 0.0],
                                       verbose=True)

        if result is None:
            print("\nFirst sample failed! Trying second sample...")
            if args.ref_file and args.hyp_file:
                if len(common_ids) > 1:
                    cut_id = sorted(common_ids)[1]
                    gt_xml = ref_dict[cut_id]
                    hyp_xml = hyp_dict[cut_id]
                    result = evaluate_single_sample(cut_id, gt_xml, hyp_xml,
                                                   collar_values=[0.25, 0.0],
                                                   verbose=True)
            else:
                samples = parse_xml_outputs_file(args.xml_file)
                if len(samples) > 1:
                    cut_id, gt_xml, hyp_xml = samples[1]
                    result = evaluate_single_sample(cut_id, gt_xml, hyp_xml,
                                                   collar_values=[0.25, 0.0],
                                                   verbose=True)

    # Evaluate full dataset
    print(f"\n\n{'='*80}")
    print("Evaluating Full Dataset")
    print("="*80)

    evaluate_full_dataset(
        xml_outputs_file=args.xml_file,
        collar_values=[0.0],
        ref_file=args.ref_file,
        hyp_file=args.hyp_file,
        output_failed_samples=args.output_failed,
        filter_cut_ids_file=args.filter_cut_ids
    )


if __name__ == "__main__":
    main()
