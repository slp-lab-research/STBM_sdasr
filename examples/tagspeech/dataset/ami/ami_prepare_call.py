# Replace /path/to/... with local paths; relative paths start at examples/tagspeech.
from ami_dataset_download import prepare_ami

prepare_ami(
    data_dir="dataset/ami",
    output_dir="AMI_manifests",
    mic="sdm",
    partition="full-corpus",
)
