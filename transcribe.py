

import whisper
import subprocess
import os
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
CLIP_DURATION  = 30       # seconds to extract
WHISPER_MODEL  = "base"   # use "large-v3" for better accuracy later


def extract_clip(audio_path: str, end_time: float, duration: int = CLIP_DURATION) -> str:
    """
    Extract a clip from audio_path ending at end_time.
    Returns path to the extracted WAV clip.
    """
    start_time = max(0, end_time - duration)

    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    clip_path = str(temp_dir / "clip.wav")

    command = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ss", str(start_time),
        "-t",  str(duration),
        "-ar", "16000",
        "-ac", "1",
        clip_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")

    print(f"Extracted clip: {start_time:.1f}s to {end_time:.1f}s")
    return clip_path


def transcribe_clip(clip_path: str, model=None) -> str:
    """
    Transcribe a WAV clip using Whisper.
    Returns the transcript as a string.
    """
    if model is None:
        print(f"Loading Whisper model ({WHISPER_MODEL})...")
        model = whisper.load_model(WHISPER_MODEL)

    print("Transcribing...")
    result = model.transcribe(
        clip_path,
        language="en",
        verbose=False,
    )

    transcript = result["text"].strip()
    print(f"Transcript: {transcript[:100]}...")
    return transcript


def get_transcript_at(audio_path: str, timestamp: float, model=None) -> str:
    """
    Main entry point.
    Given an audio file and a timestamp (in seconds), extracts the last 30s
    and returns the transcript.
    """
    clip_path  = extract_clip(audio_path, end_time=timestamp)
    transcript = transcribe_clip(clip_path, model=model)
    return transcript


# ── Standalone test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    audio_file = r"data\raw_videos\test_lecture.mp3"
    timestamp  = 60.0   # Simulate button press at 60 seconds

    if not Path(audio_file).exists():
        print(f"Audio file not found: {audio_file}")
        sys.exit(1)

    transcript = get_transcript_at(audio_file, timestamp)
    print(f"\n── Full Transcript ──\n{transcript}")