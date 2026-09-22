# Replace /path/to/... with local paths; relative paths start at examples/tagspeech.
from ami_dataset_download import download_ami

download_ami(
    target_dir="dataset/ami",
    mic="sdm",  # use "ihm", "ihm-mix", "sdm", "mdm", or "mdm8-bf"
)
