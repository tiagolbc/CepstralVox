# audio_io.py

from __future__ import annotations

import os
import uuid
import shutil
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


SUPPORTED_AUDIO_FORMATS = (
    ".wav",
    ".flac",
    ".aiff",
    ".aif",
    ".ogg",
    ".mp3",
    ".m4a",
    ".aac",
)


_TEMP_AUDIO_DIR = Path(tempfile.gettempdir()) / "cepstralvox_audio_temp"
_TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def is_supported_audio_file(file_path: str | os.PathLike) -> bool:
    """
    Returns True if the selected file has a supported audio extension.
    """
    return Path(file_path).suffix.lower() in SUPPORTED_AUDIO_FORMATS


def load_audio_as_mono(file_path: str | os.PathLike):
    """
    Loads an audio file and returns mono audio data and sample rate.

    This function supports any format readable by soundfile/libsndfile.
    For MP3/M4A support, the user's Python environment must have compatible
    backend support. If not, the function raises a clear error.
    """
    file_path = str(file_path)

    try:
        audio, sr = sf.read(file_path, always_2d=False)
    except Exception as exc:
        raise RuntimeError(
            "Could not read this audio file.\n\n"
            "CepstralVox can load WAV, FLAC, AIFF, OGG and other formats "
            "supported by your local audio backend. For MP3/M4A, you may need "
            "libsndfile/FFmpeg support depending on your environment.\n\n"
            f"Original error: {exc}"
        )

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = np.asarray(audio, dtype=np.float32)

    return audio, int(sr)


def convert_to_temp_mono_wav(
    file_path: str | os.PathLike,
    target_sr: int | None = None,
) -> str:
    """
    Converts any supported input audio file to a temporary mono WAV file.

    This does not change the original file.

    Parameters
    ----------
    file_path:
        Original audio file path.
    target_sr:
        Optional target sample rate. If None, keeps the original sample rate.

    Returns
    -------
    str
        Path to temporary mono WAV file.
    """
    file_path = str(file_path)

    if not is_supported_audio_file(file_path):
        raise ValueError(
            f"Unsupported audio format: {Path(file_path).suffix}\n"
            f"Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
        )

    audio, sr = load_audio_as_mono(file_path)

    if target_sr is not None and target_sr != sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception as exc:
            raise RuntimeError(
                "Could not resample the audio file. Install librosa or keep "
                "the original sample rate.\n\n"
                f"Original error: {exc}"
            )

    temp_name = f"cepstralvox_{uuid.uuid4().hex}.wav"
    temp_path = _TEMP_AUDIO_DIR / temp_name

    sf.write(str(temp_path), audio, sr, subtype="PCM_16")

    return str(temp_path)


def cleanup_temp_audio_files():
    """
    Removes temporary audio files created by CepstralVox.
    """
    if _TEMP_AUDIO_DIR.exists():
        try:
            shutil.rmtree(_TEMP_AUDIO_DIR)
        except Exception:
            pass

    _TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)