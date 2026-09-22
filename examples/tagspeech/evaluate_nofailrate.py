#!/usr/bin/env python3
"""Evaluate every reference cut; invalid hypotheses are scored as empty output.

Unlike evaluate.py, this evaluator never excludes a hypothesis and does not
compute or report a failure rate. A malformed or missing hypothesis receives:
all-reference-token deletions, all-reference-speech misses, and zero speaker
count credit.
"""

from __future__ import annotations

import argparse
import decimal
import logging
from pathlib import Path

from meeteval.der.md_eval import DiaErrorRate

from evaluate import (
    evaluate_single_sample,
    normalize_text,
    parse_separate_xml_files,
    parse_xml_outputs_file,
    parse_xml_to_segments,
    segments_to_cpwer_format,
)


def reference_token_count(segments) -> int:
    return sum(len(text.split()) for text in segments_to_cpwer_format(segments))


def reference_speaker_time(segments) -> float:
    return sum(max(0.0, segment.end_time - segment.start_time) for segment in segments)


def empty_der(segments) -> DiaErrorRate:
    scored_time = decimal.Decimal(str(reference_speaker_time(segments)))
    return DiaErrorRate(
        error_rate=None,
        scored_speaker_time=scored_time,
        missed_speaker_time=scored_time,
        falarm_speaker_time=decimal.Decimal("0"),
        speaker_error_time=decimal.Decimal("0"),
    )


def has_cjk(text: str) -> bool:
    return any(
        "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
        for character in text
    )


class Report:
    def __init__(self):
        self.lines = []

    def add(self, text=""):
        text = str(text)
        self.lines.append(text)
        print(text)

    def save(self, path):
        if path:
            path = Path(path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
            print(f"Report saved to: {path}")


def load_samples(args):
    selected_ids = None
    if args.filter_cut_ids:
        selected_ids = {
            line.strip()
            for line in Path(args.filter_cut_ids).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    if args.xml_file:
        samples = parse_xml_outputs_file(args.xml_file)
        if selected_ids is not None:
            samples = [sample for sample in samples if sample[0] in selected_ids]
        return samples

    references, hypotheses = parse_separate_xml_files(args.ref_file, args.hyp_file)
    if selected_ids is not None:
        references = {key: value for key, value in references.items() if key in selected_ids}
    # Reference IDs define the evaluation set. Missing hypotheses become empty.
    return [
        (cut_id, references[cut_id], hypotheses.get(cut_id, ""))
        for cut_id in sorted(references)
    ]


def evaluate(args):
    samples = load_samples(args)
    if not samples:
        raise ValueError("No reference samples were found for evaluation")

    totals = {
        "cp_errors": 0,
        "cp_length": 0,
        "cp_ins": 0,
        "cp_del": 0,
        "cp_sub": 0,
        "global_errors": 0,
        "global_length": 0,
        "global_ins": 0,
        "global_del": 0,
        "global_sub": 0,
        "count_correct": 0,
        "count_abs_error": 0,
        "gender_correct_time": 0.0,
        "gender_total_time": 0.0,
        "gender_correct_pairs": 0,
        "gender_total_pairs": 0,
    }
    total_der = None
    cjk_dataset = False

    for index, (cut_id, reference_xml, hypothesis_xml) in enumerate(samples, 1):
        if index % 100 == 0:
            print(f"Processing {index}/{len(samples)}...")

        reference_segments = parse_xml_to_segments(reference_xml)
        if not reference_segments:
            raise ValueError(
                f"Reference XML is invalid for {cut_id}; references cannot be assigned a model score"
            )
        cjk_dataset = cjk_dataset or has_cjk(reference_xml)
        hypothesis_segments = parse_xml_to_segments(hypothesis_xml) if hypothesis_xml.strip() else []

        result = None
        if hypothesis_segments:
            result = evaluate_single_sample(
                cut_id,
                reference_xml,
                hypothesis_xml,
                collar_values=[0.0],
                verbose=False,
            )

        if result is None:
            # Zero predicted output: all reference tokens are deletions.
            token_count = reference_token_count(reference_segments)
            totals["cp_errors"] += token_count
            totals["cp_length"] += token_count
            totals["cp_del"] += token_count
            totals["global_errors"] += token_count
            totals["global_length"] += token_count
            totals["global_del"] += token_count

            der = empty_der(reference_segments)
            total_der = der if total_der is None else total_der + der

            reference_speakers = len({segment.speaker_id for segment in reference_segments})
            totals["count_abs_error"] += reference_speakers

            # A zero prediction has zero correct gender time; reference speaker
            # time remains in the denominator so it receives zero credit.
            totals["gender_total_time"] += reference_speaker_time(reference_segments)
            totals["gender_total_pairs"] += reference_speakers
            continue

        cp = result["cpwer"]
        totals["cp_errors"] += cp.errors
        totals["cp_length"] += cp.length
        totals["cp_ins"] += cp.insertions
        totals["cp_del"] += cp.deletions
        totals["cp_sub"] += cp.substitutions

        global_wer = result["global_wer"]
        totals["global_errors"] += global_wer.errors
        totals["global_length"] += global_wer.length
        totals["global_ins"] += global_wer.insertions
        totals["global_del"] += global_wer.deletions
        totals["global_sub"] += global_wer.substitutions

        if result.get("der_failed") or 0.0 not in result["der"]:
            der = empty_der(reference_segments)
        else:
            der = result["der"][0.0]
        total_der = der if total_der is None else total_der + der

        gender = result["gender"]
        totals["gender_correct_time"] += gender["correct_time"]
        totals["gender_total_time"] += gender["total_time"]
        totals["gender_correct_pairs"] += gender["num_correct_pairs"]
        totals["gender_total_pairs"] += gender["num_aligned_pairs"]

        count = result["count"]
        totals["count_correct"] += count["exact_match"]
        totals["count_abs_error"] += count["absolute_error"]

    unit = "CER" if cjk_dataset else "WER"
    cp_rate = totals["cp_errors"] / totals["cp_length"] if totals["cp_length"] else 0.0
    global_rate = (
        totals["global_errors"] / totals["global_length"]
        if totals["global_length"] else 0.0
    )
    sca = totals["count_correct"] / len(samples)
    count_mae = totals["count_abs_error"] / len(samples)

    report = Report()
    report.add("=" * 80)
    report.add("AGGREGATE RESULTS — ALL REFERENCE SAMPLES INCLUDED")
    report.add("=" * 80)
    report.add()
    report.add("--- Processing Statistics ---")
    report.add(f"Total samples evaluated: {len(samples)}")
    report.add("Scoring policy: invalid or missing hypotheses are empty predictions")
    report.add()
    report.add(f"--- cp{unit} (Concatenated Minimum-Permutation {unit}) ---")
    report.add(f"  cp{unit}: {cp_rate:.2%}")
    report.add(f"  Total Errors: {totals['cp_errors']}")
    report.add(f"  Total Length: {totals['cp_length']}")
    report.add(f"  Insertions: {totals['cp_ins']}")
    report.add(f"  Deletions: {totals['cp_del']}")
    report.add(f"  Substitutions: {totals['cp_sub']}")
    report.add()
    report.add(f"--- Global {unit} (Speaker-Agnostic) ---")
    report.add(f"  g{unit}: {global_rate:.2%}")
    report.add(f"  Total Errors: {totals['global_errors']}")
    report.add(f"  Total Length: {totals['global_length']}")
    report.add(f"  Insertions: {totals['global_ins']}")
    report.add(f"  Deletions: {totals['global_del']}")
    report.add(f"  Substitutions: {totals['global_sub']}")
    report.add()
    report.add("--- DER (Diarization Error Rate, collar=0.0s) ---")
    if total_der is not None and float(total_der.scored_speaker_time) > 0:
        scored = float(total_der.scored_speaker_time)
        report.add(f"  DER: {float(total_der.error_rate):.2%}")
        report.add(f"  Miss: {float(total_der.missed_speaker_time) / scored:.2%}")
        report.add(f"  False Alarm: {float(total_der.falarm_speaker_time) / scored:.2%}")
        report.add(f"  Speaker Confusion: {float(total_der.speaker_error_time) / scored:.2%}")
        report.add(f"  Total Scored Speaker Time: {scored:.2f}s")
    else:
        report.add("  No scored reference speaker time")
    report.add()
    report.add("--- Speaker Count Metrics ---")
    report.add(f"  SCA: {sca:.2%} ({totals['count_correct']}/{len(samples)} samples)")
    report.add(f"  Speaker Count MAE: {count_mae:.2f}")
    report.add()
    report.add("--- Gender Accuracy (time-weighted) ---")
    if totals["gender_total_time"]:
        gender_accuracy = totals["gender_correct_time"] / totals["gender_total_time"]
        report.add(f"  Gender Accuracy: {gender_accuracy:.2%}")
        report.add(
            f"  Correct Time: {totals['gender_correct_time']:.2f}s / "
            f"{totals['gender_total_time']:.2f}s"
        )
    else:
        report.add("  No gender-scored reference time")
    report.add()
    report.add("=" * 80)
    report.save(args.output_report)


def main():
    # meeteval emits several UEM/filename warnings per cut; keep the aggregate
    # report readable while still allowing genuine errors to surface.
    logging.disable(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml_file")
    parser.add_argument("--ref_file")
    parser.add_argument("--hyp_file")
    parser.add_argument("--filter_cut_ids")
    parser.add_argument("--output_report", help="Optional path for the aggregate report table")
    args = parser.parse_args()
    if bool(args.xml_file) == bool(args.ref_file or args.hyp_file):
        parser.error("Use either --xml_file or both --ref_file and --hyp_file")
    if not args.xml_file and not (args.ref_file and args.hyp_file):
        parser.error("Both --ref_file and --hyp_file are required together")
    evaluate(args)


if __name__ == "__main__":
    main()
