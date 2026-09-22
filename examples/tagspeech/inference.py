import argparse
from pathlib import Path

import torch

from model import TagSpeechModel
from utils.xml_utils import xml_to_json


def get_args():
    parser = argparse.ArgumentParser(description="Run TagSpeech inference on WAV files.")
    parser.add_argument("wav_files", nargs="+", help="One or more input WAV files.")
    parser.add_argument("--model", required=True, help="Local model path or HF model ID.")
    parser.add_argument("--max-new-tokens", type=int, default=800)
    parser.add_argument("--num-beams", type=int, default=1)
    return parser.parse_args()


def main():
    args = get_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run inference on a CUDA-enabled machine."
        )

    wav_files = [str(Path(path).expanduser().resolve()) for path in args.wav_files]
    missing = [path for path in wav_files if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Input WAV file(s) not found: {missing}")

    device = torch.device("cuda:0")
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"Loading model: {args.model}")
    model = TagSpeechModel.from_pretrained(args.model).to(device)
    model.eval()

    audio_token = model.config.audio_token
    user_content = (
        f"<audio>{audio_token}</audio>"
        if getattr(model.config, "use_segment_speaker_kernel", False)
        else f"<text>{audio_token}</text>\n<speaker>{audio_token}</speaker>"
    )
    messages = [
        [
            {
                "role": "user",
                "content": user_content,
            }
        ]
        for _ in wav_files
    ]

    outputs = model.generate(
        wav_files,
        messages,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=False,
    )

    for i, output in enumerate(outputs, 1):
        print(
            f"\n{'=' * 80}\nOutput {i}/{len(outputs)} - XML:\n"
            f"{'=' * 80}\n{output}\n{'=' * 80}"
        )

        json_output = xml_to_json(output)
        if json_output:
            print(
                f"\nOutput {i}/{len(outputs)} - JSON:\n"
                f"{'=' * 80}\n{json_output}\n{'=' * 80}"
            )
        else:
            print(
                f"\nWarning: Output {i} could not be parsed as valid XML\n"
                f"{'=' * 80}"
            )


if __name__ == "__main__":
    main()
