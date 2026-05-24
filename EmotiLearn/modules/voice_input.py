# ============================================================
# modules/voice_input.py
# Voice capture module — records from microphone and saves WAV
#
# Design decisions:
#   - Uses sounddevice (preferred) with PyAudio as fallback
#   - Always outputs 16kHz mono WAV (Whisper's expected format)
#   - Supports both fixed-duration and silence-based recording
#   - Handles file upload path too (for Streamlit/API use)
# ============================================================

import io
import wave
import time
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Lazy imports — only loaded when actually recording
_sounddevice = None
_pyaudio = None

from config.settings import settings
from modules.logger import logger


def _get_sounddevice():
    """Lazy-load sounddevice."""
    global _sounddevice
    if _sounddevice is None:
        import sounddevice as sd
        _sounddevice = sd
    return _sounddevice


def _get_pyaudio():
    """Lazy-load PyAudio as fallback."""
    global _pyaudio
    if _pyaudio is None:
        import pyaudio
        _pyaudio = pyaudio
    return _pyaudio


class AudioRecorder:
    """
    Records audio from the system microphone.

    Supports two backends:
      1. sounddevice  (preferred — simpler API, numpy native)
      2. PyAudio      (fallback — broader hardware support)

    Always outputs 16kHz mono PCM WAV for Whisper compatibility.
    """

    def __init__(
        self,
        sample_rate: int = None,
        channels: int = None,
        chunk_size: int = None,
    ):
        self.sample_rate = sample_rate or settings.sample_rate   # 16000 Hz
        self.channels = channels or settings.audio_channels      # 1 (mono)
        self.chunk_size = chunk_size or settings.chunk_size      # 1024

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        duration: float = None,
        output_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Record for a fixed duration.

        Args:
            duration:    Seconds to record (default from settings)
            output_path: Where to save .wav file (uses temp file if None)

        Returns:
            (audio_array [float32, shape=(N,)], wav_file_path)
        """
        duration = duration or settings.record_seconds
        output_path = output_path or self._temp_wav_path()

        logger.info(f"Recording {duration}s at {self.sample_rate}Hz...")

        try:
            audio = self._record_sounddevice(duration)
        except Exception as e:
            logger.warning(f"sounddevice failed ({e}), trying PyAudio...")
            try:
                audio = self._record_pyaudio(duration)
            except Exception as e2:
                raise RuntimeError(
                    f"Both audio backends failed.\n"
                    f"sounddevice: {e}\nPyAudio: {e2}\n"
                    "Please check your microphone is connected."
                )

        audio = self._normalise(audio)
        self._save_wav(audio, output_path)
        logger.info(f"Saved recording to {output_path}")

        return audio, output_path

    def record_with_silence_detection(
        self,
        max_duration: float = 10.0,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.5,
        output_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Record until user stops speaking (auto silence detection).

        Stops when silence_duration seconds of silence is detected,
        or max_duration seconds is reached.

        Returns:
            (audio_array, wav_file_path)
        """
        output_path = output_path or self._temp_wav_path()
        sd = _get_sounddevice()

        logger.info(f"Recording (auto-stop on silence, max={max_duration}s)...")

        frames = []
        silence_counter = 0
        block_duration = self.chunk_size / self.sample_rate
        silence_blocks = int(silence_duration / block_duration)

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.chunk_size,
        ) as stream:
            start = time.time()
            while time.time() - start < max_duration:
                block, _ = stream.read(self.chunk_size)
                block_mono = block[:, 0] if block.ndim > 1 else block
                frames.append(block_mono.copy())

                rms = float(np.sqrt(np.mean(block_mono ** 2)))
                if rms < silence_threshold:
                    silence_counter += 1
                    if silence_counter >= silence_blocks:
                        logger.debug("Silence detected — stopping recording.")
                        break
                else:
                    silence_counter = 0

        if not frames:
            raise RuntimeError("No audio recorded. Check microphone.")

        audio = np.concatenate(frames, axis=0)
        audio = self._normalise(audio)
        self._save_wav(audio, output_path)

        elapsed = len(audio) / self.sample_rate
        logger.info(f"Recorded {elapsed:.1f}s to {output_path}")
        return audio, output_path

    # ------------------------------------------------------------------
    # Private: recording backends
    # ------------------------------------------------------------------

    def _record_sounddevice(self, duration: float) -> np.ndarray:
        """Record using sounddevice (preferred)."""
        sd = _get_sounddevice()
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
        )
        sd.wait()
        return audio.flatten()

    def _record_pyaudio(self, duration: float) -> np.ndarray:
        """Record using PyAudio (fallback)."""
        pa = _get_pyaudio()
        p = pa.PyAudio()
        stream = p.open(
            format=pa.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        frames = []
        n_chunks = int(self.sample_rate / self.chunk_size * duration)
        for _ in range(n_chunks):
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.float32))
        stream.stop_stream()
        stream.close()
        p.terminate()
        return np.concatenate(frames)

    # ------------------------------------------------------------------
    # Private: helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(audio: np.ndarray) -> np.ndarray:
        """Normalise audio to float32 [-1.0, 1.0]."""
        audio = audio.astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio

    def _save_wav(self, audio: np.ndarray, path: str) -> None:
        """Save float32 numpy array as 16-bit PCM WAV."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pcm = (audio * 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())

    @staticmethod
    def _temp_wav_path() -> str:
        """Return path for a new temporary WAV file."""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        return tmp.name


# ------------------------------------------------------------------
# Module-level utility functions (used by API + Streamlit)
# ------------------------------------------------------------------

def load_audio_file(
    file_path: str,
    target_sr: int = None,
) -> Tuple[np.ndarray, int]:
    """
    Load any audio file (wav, mp3, flac, ogg) and resample to target_sr.
    Uses librosa under the hood.

    Returns: (audio float32, sample_rate)
    """
    import librosa

    target_sr = target_sr or settings.sample_rate
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        logger.debug(f"Loaded audio: {file_path} ({len(audio)/sr:.2f}s @ {sr}Hz)")
        return audio, sr
    except Exception as e:
        raise ValueError(f"Failed to load audio file '{file_path}': {e}")


def load_audio_bytes(
    audio_bytes: bytes,
    target_sr: int = None,
) -> Tuple[np.ndarray, int]:
    """
    Load audio from raw bytes (used by FastAPI upload endpoint).
    Returns: (audio float32, sample_rate)
    """
    import soundfile as sf
    import librosa

    target_sr = target_sr or settings.sample_rate
    buf = io.BytesIO(audio_bytes)

    try:
        audio, sr = sf.read(buf, dtype="float32", always_2d=False)
    except Exception:
        buf.seek(0)
        audio, sr = librosa.load(buf, sr=None, mono=True)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio.astype(np.float32), sr


def save_audio_bytes(audio_bytes: bytes, output_path: Optional[str] = None) -> str:
    """Save uploaded audio bytes to a WAV file. Returns path."""
    output_path = output_path or AudioRecorder._temp_wav_path()
    audio, sr = load_audio_bytes(audio_bytes)
    recorder = AudioRecorder(sample_rate=sr)
    recorder._save_wav(audio, output_path)
    return output_path


def list_audio_devices() -> None:
    """Print all available audio input devices (debugging helper)."""
    try:
        sd = _get_sounddevice()
        print("\nAvailable Audio Devices:")
        print(sd.query_devices())
        print(f"\nDefault input: {sd.query_devices(kind='input')['name']}")
    except Exception as e:
        print(f"Could not list devices: {e}")
