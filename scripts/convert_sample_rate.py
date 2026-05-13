#!/usr/bin/env python3
"""Convert audio files to 16000 Hz sample rate."""

import librosa
import soundfile as sf
from pathlib import Path


def convert_to_16k(input_path: Path, target_sr: int = 16000) -> None:
    """Convert audio file to target sample rate.

    Args:
        input_path: Path to input audio file
        target_sr: Target sample rate (default: 16000)
    """
    # Load audio with original sample rate
    audio, sr = librosa.load(str(input_path), sr=None)

    # Resample if needed
    if sr != target_sr:
        print(f"  Converting {input_path.name}: {sr} Hz -> {target_sr} Hz")
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        # Backup original
        backup_path = input_path.with_suffix('.wav.bak')
        if not backup_path.exists():
            import shutil
            shutil.copy(input_path, backup_path)
            print(f"  ✓ Backup: {backup_path.name}")

        # Write converted file
        sf.write(str(input_path), audio_16k, target_sr)
        print(f"  ✓ Converted: {input_path.name}")
    else:
        print(f"  Already {target_sr} Hz: {input_path.name}")


def main():
    """Convert all audio files in specified directories."""
    base_dir = Path(__file__).parent.parent / "asset"

    speakers = ["john", "xixi", "frank"]

    for speaker in speakers:
        speaker_dir = base_dir / speaker
        if not speaker_dir.exists():
            print(f"Speaker directory not found: {speaker_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {speaker.upper()}")
        print(f"{'='*60}")

        # Find all wav files
        for wav_file in speaker_dir.rglob("*.wav"):
            convert_to_16k(wav_file)


if __name__ == "__main__":
    main()
