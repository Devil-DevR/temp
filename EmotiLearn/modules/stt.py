# ============================================================
# modules/stt.py
# Speech-to-Text using OpenAI Whisper Tiny
#
# Design decisions:
#   - Whisper Tiny: 39MB, ~400-700ms on CPU — perfect for our latency budget
#   - Model cached in memory after first load (singleton pattern)
#   - fp16=False always (CPU doesn't support half precision inference)
#   - Supports both file path and numpy array input
#   - Returns transcript + language + confidence metadata
# ============================================================

import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field

import numpy as np

from config.settings import settings
from modules.logger import logger


@dataclass
class TranscriptionResult:
    """
    Structured result from Whisper transcription.

    Attributes:
        text:          Cleaned transcript string
        language:      Detected language code (e.g. 'en')
        segments:      List of timed segments [{start, end, text}]
        duration_s:    Length of audio processed (seconds)
        latency_ms:    Time taken for inference (milliseconds)
        model_size:    Whisper model variant used
        is_empty:      True if no speech detected
    """
    text: str
    language: str
    segments: list = field(default_factory=list)
    duration_s: float = 0.0
    latency_ms: float = 0.0
    model_size: str = "tiny"
    is_empty: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "segments": self.segments,
            "duration_s": round(self.duration_s, 2),
            "latency_ms": round(self.latency_ms, 1),
            "model_size": self.model_size,
            "is_empty": self.is_empty,
        }


class WhisperSTT:
    """
    Singleton wrapper around OpenAI Whisper.

    Key optimizations:
      - Model loaded once and cached in RAM
      - Thread-safe with a loading lock
      - fp16 disabled for CPU inference
      - Audio preprocessed to Whisper's expected format before inference

    Usage:
        stt = WhisperSTT.get_instance()
        result = stt.transcribe("path/to/audio.wav")
        print(result.text)
    """

    _instance: Optional["WhisperSTT"] = None
    _lock = threading.Lock()

    def __init__(self, model_size: str = None):
        self.model_size = model_size or settings.whisper_model_size
        self._model = None
        self._model_loaded = False
        logger.info(f"WhisperSTT initialised (model={self.model_size})")

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls, model_size: str = None) -> "WhisperSTT":
        """
        Return the shared WhisperSTT instance.
        Creates it on first call (thread-safe).
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-check inside lock
                    cls._instance = cls(model_size)
        return cls._instance

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load Whisper model into memory.
        Called automatically on first transcription.
        Model is cached in models/whisper/ to avoid re-downloading.
        """
        if self._model_loaded:
            return

        import whisper

        cache_dir = Path(settings.models_cache_dir) / "whisper"
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading Whisper {self.model_size} model...")
        t0 = time.time()

        self._model = whisper.load_model(
            self.model_size,
            device="cpu",
            download_root=str(cache_dir),
        )
        self._model_loaded = True

        elapsed = (time.time() - t0) * 1000
        logger.info(f"Whisper {self.model_size} loaded in {elapsed:.0f}ms")

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_input: Union[str, np.ndarray],
        language: str = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio_input: Either a file path (str) or numpy float32 array
                         at 16kHz sample rate
            language:    Force language code (e.g. 'en') or None for auto

        Returns:
            TranscriptionResult with text, language, timing info
        """
        self.load_model()

        t0 = time.time()

        # Prepare audio
        if isinstance(audio_input, str):
            audio_path = audio_input
            audio_array = self._load_and_preprocess(audio_path)
            duration_s = len(audio_array) / settings.sample_rate
        elif isinstance(audio_input, np.ndarray):
            audio_array = self._preprocess_array(audio_input)
            duration_s = len(audio_array) / settings.sample_rate
            audio_path = None
        else:
            raise TypeError(f"audio_input must be str or np.ndarray, got {type(audio_input)}")

        logger.debug(f"Transcribing {duration_s:.1f}s of audio...")

        # Whisper inference
        transcribe_options = {
            "fp16": False,           # CPU only — no half precision
            "language": language,    # None = auto-detect
            "verbose": False,
            "condition_on_previous_text": False,  # Faster for short clips
        }

        result = self._model.transcribe(audio_array, **transcribe_options)

        latency_ms = (time.time() - t0) * 1000
        text = result["text"].strip()

        transcription = TranscriptionResult(
            text=text,
            language=result.get("language", "en"),
            segments=[
                {
                    "start": round(s["start"], 2),
                    "end": round(s["end"], 2),
                    "text": s["text"].strip(),
                }
                for s in result.get("segments", [])
            ],
            duration_s=duration_s,
            latency_ms=latency_ms,
            model_size=self.model_size,
            is_empty=(len(text) == 0),
        )

        logger.info(
            f"Transcribed in {latency_ms:.0f}ms | "
            f"lang={transcription.language} | "
            f"text='{text[:80]}{'...' if len(text) > 80 else ''}'"
        )

        return transcription

    def transcribe_with_fallback(
        self,
        audio_input: Union[str, np.ndarray],
        min_length_chars: int = 3,
    ) -> TranscriptionResult:
        """
        Transcribe with automatic retry using 'base' model if tiny gives
        empty or very short result.

        Useful for noisy audio or non-native speakers.
        """
        result = self.transcribe(audio_input)

        if result.is_empty or len(result.text) < min_length_chars:
            logger.warning(
                f"Tiny model returned short/empty result: '{result.text}'. "
                "Retrying with base model..."
            )
            fallback_stt = WhisperSTT(model_size="base")
            result = fallback_stt.transcribe(audio_input)

        return result

    # ------------------------------------------------------------------
    # Audio preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _load_and_preprocess(file_path: str) -> np.ndarray:
        """
        Load audio file and convert to Whisper-expected format:
        - float32
        - 16kHz sample rate
        - Mono channel
        """
        import whisper

        audio = whisper.load_audio(file_path)  # Handles any format via ffmpeg
        audio = whisper.pad_or_trim(audio)     # Pad/trim to 30s (Whisper's window)
        return audio

    @staticmethod
    def _preprocess_array(audio: np.ndarray) -> np.ndarray:
        """
        Preprocess a numpy audio array to Whisper's expected format.
        """
        import whisper

        # Ensure float32
        audio = audio.astype(np.float32)

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        # Pad or trim to Whisper's 30-second window
        audio = whisper.pad_or_trim(audio)

        return audio

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for logging/debugging."""
        if not self._model_loaded:
            return {"loaded": False, "model_size": self.model_size}

        dims = self._model.dims
        return {
            "loaded": True,
            "model_size": self.model_size,
            "n_mels": dims.n_mels,
            "n_audio_ctx": dims.n_audio_ctx,
            "n_audio_state": dims.n_audio_state,
            "n_audio_head": dims.n_audio_head,
            "n_audio_layer": dims.n_audio_layer,
            "n_vocab": dims.n_vocab,
            "n_text_ctx": dims.n_text_ctx,
        }


# ------------------------------------------------------------------
# Convenience function for one-shot transcription
# (used by API routes and tests)
# ------------------------------------------------------------------

def transcribe_audio(
    audio_input: Union[str, np.ndarray],
    model_size: str = None,
    language: str = None,
) -> TranscriptionResult:
    """
    One-shot transcription using the global WhisperSTT singleton.

    Args:
        audio_input: File path or numpy float32 array at 16kHz
        model_size:  Override default model ('tiny', 'base', 'small')
        language:    Force language or None for auto-detect

    Returns:
        TranscriptionResult
    """
    stt = WhisperSTT.get_instance(model_size)
    return stt.transcribe(audio_input, language=language)
