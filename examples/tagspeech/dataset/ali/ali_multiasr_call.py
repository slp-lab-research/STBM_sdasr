# Replace /path/to/... with local paths; relative paths start at examples/tagspeech.
from pathlib import Path

from lhotse import CutSet, RecordingSet, SupervisionSet


manifest_dir = Path("AliMeeting_manifests")
mic = "far"

for split in ["train", "eval", "test"]:
    recordings = RecordingSet.from_file(
        manifest_dir / f"alimeeting-{mic}_recordings_{split}.jsonl.gz"
    )
    supervisions = SupervisionSet.from_file(
        manifest_dir / f"alimeeting-{mic}_supervisions_{split}.jsonl.gz"
    )
    cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
    cuts = cuts.trim_to_supervision_groups(max_pause=0.0)
    cuts.to_file(manifest_dir / f"alimeeting-{mic}_{split}_multiASR.jsonl.gz")
