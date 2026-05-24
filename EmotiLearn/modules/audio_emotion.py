# ============================================================
# modules/audio_emotion.py
# Audio Emotion Detection Pipeline
#
# Architecture:
#   Raw WAV → librosa MFCC (40 coeffs + delta + delta-delta)
#   → shape (3, 40, T) → Small CNN → Softmax → emotion probs
#
# Design decisions:
#   - 3-channel input: MFCC + Δ + ΔΔ (like RGB for audio)
#   - Fixed time dimension via padding/truncation to max_len frames
#   - CNN chosen over RNN for speed on CPU (~30ms inference)
#   - Batch norm + dropout for regularization on small datasets
#   - Singleton inference class (model loaded once, cached)
# ============================================================

import time
import threading
from pathlib import Path
from typing import Optional, Dict, Union, Tuple
from dataclasses import dataclass

import numpy as np

from config.settings import settings
from modules.logger import logger


# ---------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------

def extract_mfcc_features(
    audio: np.ndarray,
    sample_rate: int = None,
    n_mfcc: int = None,
    n_fft: int = None,
    hop_length: int = None,
    max_len: int = 128,
) -> np.ndarray:
    """
    Extract 3-channel MFCC feature map from raw audio.

    Channels:
        Channel 0: MFCC coefficients          (static, spectral shape)
        Channel 1: Delta MFCC                 (rate of change)
        Channel 2: Delta-Delta MFCC           (acceleration)

    Why 3 channels?
        Mirrors the RGB convention for 2D CNNs. Delta features capture
        temporal dynamics — crucial for distinguishing confused (slow,
        uncertain speech) from engaged (fast, energetic speech).

    Args:
        audio:       Float32 numpy array, mono, any sample rate
        sample_rate: Audio sample rate (default: settings.sample_rate)
        n_mfcc:      Number of MFCC coefficients (default: 40)
        n_fft:       FFT window size (default: 2048)
        hop_length:  Hop size between frames (default: 512)
        max_len:     Fixed output time dimension (pad or trim)

    Returns:
        np.ndarray of shape (3, n_mfcc, max_len) — float32
    """
    import librosa

    sr = sample_rate or settings.sample_rate
    n_mfcc = n_mfcc or settings.n_mfcc
    n_fft = n_fft or settings.n_fft
    hop_length = hop_length or settings.hop_length

    # Ensure float32 mono
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    # Pad very short audio to avoid librosa errors
    min_samples = n_fft
    if len(audio) < min_samples:
        audio = np.pad(audio, (0, min_samples - len(audio)))

    # ---- Extract MFCC + deltas ----
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )                                            # shape: (n_mfcc, T)

    delta = librosa.feature.delta(mfcc)          # 1st derivative
    delta2 = librosa.feature.delta(mfcc, order=2)  # 2nd derivative

    # ---- Standardise each channel independently (zero mean, unit var) ----
    def standardise(x):
        mean, std = x.mean(), x.std()
        return (x - mean) / (std + 1e-8)

    mfcc = standardise(mfcc)
    delta = standardise(delta)
    delta2 = standardise(delta2)

    # ---- Stack into 3-channel feature map: (3, n_mfcc, T) ----
    features = np.stack([mfcc, delta, delta2], axis=0)  # (3, 40, T)

    # ---- Pad or truncate to fixed time dimension ----
    T = features.shape[2]
    if T < max_len:
        pad_width = max_len - T
        features = np.pad(features, ((0, 0), (0, 0), (0, pad_width)))
    else:
        features = features[:, :, :max_len]

    return features.astype(np.float32)


def extract_features_from_file(
    file_path: str,
    max_len: int = 128,
) -> np.ndarray:
    """
    Load an audio file and extract MFCC features.
    Convenience wrapper around extract_mfcc_features.

    Returns: np.ndarray shape (3, n_mfcc, max_len)
    """
    from modules.voice_input import load_audio_file
    audio, sr = load_audio_file(file_path, target_sr=settings.sample_rate)
    return extract_mfcc_features(audio, sample_rate=sr, max_len=max_len)


# ---------------------------------------------------------------
# CNN Model Architecture
# ---------------------------------------------------------------

def build_audio_cnn(num_classes: int = None) -> "torch.nn.Module":
    """
    Build the small CNN for audio emotion classification.

    Architecture overview:
        Input: (B, 3, 40, 128)  — batch × channels × mfcc × time

        Block 1: Conv(3→32, 3×3) → BN → ReLU → MaxPool(2×2)  → (B, 32, 20, 64)
        Block 2: Conv(32→64, 3×3) → BN → ReLU → MaxPool(2×2) → (B, 64, 10, 32)
        Block 3: Conv(64→128, 3×3) → BN → ReLU → MaxPool(2×2) → (B, 128, 5, 16)
        Global Avg Pool                                          → (B, 128)
        FC(128 → 64) → ReLU → Dropout(0.3)
        FC(64 → num_classes)                                    → (B, 6)

    Why Global Average Pooling instead of Flatten?
        - Reduces parameters (no giant FC layer)
        - Built-in spatial invariance
        - Less prone to overfitting on small datasets

    Total parameters: ~180K  (very lightweight, loads in <10ms on CPU)
    """
    import torch
    import torch.nn as nn

    num_classes = num_classes or settings.num_classes

    class AudioEmotionCNN(nn.Module):
        def __init__(self):
            super().__init__()

            # Convolutional blocks
            self.block1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )
            self.block3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

            # Global average pooling: reduces spatial dims to scalar per channel
            self.gap = nn.AdaptiveAvgPool2d(1)

            # Classifier head
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(settings.cnn_dropout),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            # x: (B, 3, 40, 128)
            x = self.block1(x)   # (B, 32, 20, 64)
            x = self.block2(x)   # (B, 64, 10, 32)
            x = self.block3(x)   # (B, 128, 5, 16)
            x = self.gap(x)      # (B, 128, 1, 1)
            x = self.classifier(x)  # (B, num_classes)
            return x

        def predict_proba(self, x):
            """Return softmax probabilities."""
            import torch.nn.functional as F
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    return AudioEmotionCNN()


# ---------------------------------------------------------------
# Inference class (singleton, cached model)
# ---------------------------------------------------------------

@dataclass
class AudioEmotionResult:
    """Structured result from audio emotion inference."""
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float
    features_shape: tuple

    def to_dict(self):
        return {
            "emotion": self.emotion,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "latency_ms": round(self.latency_ms, 1),
        }


class AudioEmotionClassifier:
    """
    Singleton inference engine for audio emotion classification.

    Loads the trained CNN once and keeps it in memory.
    Accepts raw audio arrays or file paths.

    Usage:
        clf = AudioEmotionClassifier.get_instance()
        result = clf.predict(audio_array)
        print(result.emotion, result.confidence)
    """

    _instance: Optional["AudioEmotionClassifier"] = None
    _lock = threading.Lock()

    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.audio_cnn_model_path
        self._model = None
        self._loaded = False
        self.emotion_labels = settings.emotion_labels
        logger.info(f"AudioEmotionClassifier initialised (model={self.model_path})")

    @classmethod
    def get_instance(cls, model_path: str = None) -> "AudioEmotionClassifier":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_path)
        return cls._instance

    def load_model(self) -> None:
        """Load trained CNN weights into memory."""
        if self._loaded:
            return

        import torch

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Audio CNN model not found at: {model_path}\n"
                "Please run: python training/train_audio_cnn.py"
            )

        logger.info(f"Loading Audio CNN from {model_path}...")
        t0 = time.time()

        self._model = build_audio_cnn(num_classes=len(self.emotion_labels))
        checkpoint = torch.load(str(model_path), map_location="cpu")

        # Handle both raw state_dict and wrapped checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self._model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self._model.load_state_dict(checkpoint)

        self._model.eval()
        elapsed = (time.time() - t0) * 1000
        logger.info(f"Audio CNN loaded in {elapsed:.0f}ms")
        self._loaded = True

    def predict(
        self,
        audio_input: Union[str, np.ndarray],
        sample_rate: int = None,
    ) -> AudioEmotionResult:
        """
        Predict emotion from audio.

        Args:
            audio_input: File path (str) or numpy float32 array
            sample_rate: SR of audio array (ignored for file paths)

        Returns:
            AudioEmotionResult with emotion, confidence, probabilities
        """
        self.load_model()

        import torch

        t0 = time.time()

        # Extract features
        if isinstance(audio_input, str):
            features = extract_features_from_file(audio_input)
        elif isinstance(audio_input, np.ndarray):
            sr = sample_rate or settings.sample_rate
            features = extract_mfcc_features(audio_input, sample_rate=sr)
        else:
            raise TypeError(f"Expected str or np.ndarray, got {type(audio_input)}")

        # Add batch dimension: (3, 40, 128) → (1, 3, 40, 128)
        tensor = torch.from_numpy(features).unsqueeze(0)

        # Inference (no gradient computation needed)
        with torch.no_grad():
            probs = self._model.predict_proba(tensor)  # (1, num_classes)

        probs_np = probs.squeeze(0).numpy()  # (num_classes,)
        pred_idx = int(np.argmax(probs_np))

        latency_ms = (time.time() - t0) * 1000

        result = AudioEmotionResult(
            emotion=self.emotion_labels[pred_idx],
            confidence=float(probs_np[pred_idx]),
            probabilities={
                label: float(probs_np[i])
                for i, label in enumerate(self.emotion_labels)
            },
            latency_ms=latency_ms,
            features_shape=features.shape,
        )

        logger.debug(
            f"Audio emotion: {result.emotion} "
            f"({result.confidence:.1%}) in {latency_ms:.0f}ms"
        )
        return result

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------

def predict_audio_emotion(
    audio_input: Union[str, np.ndarray],
    model_path: str = None,
) -> AudioEmotionResult:
    """
    One-shot audio emotion prediction using the global singleton.

    Args:
        audio_input: File path or numpy float32 array at 16kHz
        model_path:  Override default model path

    Returns:
        AudioEmotionResult
    """
    clf = AudioEmotionClassifier.get_instance(model_path)
    return clf.predict(audio_input)
