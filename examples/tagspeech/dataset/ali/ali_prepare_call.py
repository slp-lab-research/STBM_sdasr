# Replace /path/to/... with local paths; relative paths start at examples/tagspeech.
from ali_dataset_download import prepare_ali_meeting

prepare_ali_meeting(
    corpus_dir="dataset/ali",
    output_dir="AliMeeting_manifests",
    mic="far",
)
