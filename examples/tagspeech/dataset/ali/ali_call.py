# Replace /path/to/... with local paths; relative paths start at examples/tagspeech.
from ali_dataset_download import download_ali_meeting

download_ali_meeting(
    target_dir="dataset/ali",
)
