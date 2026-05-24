# ============================================================
# modules/text_emotion.py
# Text Emotion Detection using fine-tuned DistilBERT
#
# Architecture:
#   Raw text → DistilBERT tokenizer → DistilBERT encoder
#   → [CLS] token embedding → Dropout → Linear(768 → 6) → Softmax
#
# Design decisions:
#   - DistilBERT (66M params) over BERT (110M) — 40% smaller, 60% faster
#   - Only the classification head is fine-tuned in early layers;
#     all layers are fine-tuned with a low LR (2e-5) for best accuracy
#   - Dynamic quantization applied post-training: Linear layers
#     converted to INT8, reducing model to ~25MB and speeding up ~1.5×
#   - Singleton pattern: tokenizer + model cached after first load
#   - max_length=128 covers >99% of student utterances (<100 words)
# ============================================================

import time
import threading
from pathlib import Path
from typing import Optional, Dict, Union, List
from dataclasses import dataclass, field

from config.settings import settings
from modules.logger import logger


# ---------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------

@dataclass
class TextEmotionResult:
    """
    Structured output from text emotion inference.

    Attributes:
        text:          Input text that was classified
        emotion:       Predicted emotion label (str)
        confidence:    Softmax probability of top prediction [0, 1]
        probabilities: Full distribution over all 6 emotion classes
        latency_ms:    Inference time in milliseconds
        token_count:   Number of tokens the input was split into
    """
    text: str
    emotion: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    token_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "emotion": self.emotion,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "latency_ms": round(self.latency_ms, 1),
            "token_count": self.token_count,
        }

    @property
    def is_low_confidence(self) -> bool:
        """Flag when model is uncertain (confidence below threshold)."""
        return self.confidence < 0.45


# ---------------------------------------------------------------
# DistilBERT classifier model definition
# ---------------------------------------------------------------

def build_distilbert_classifier(
    base_model_name: str = None,
    num_classes: int = None,
    dropout: float = 0.3,
):
    """
    Build DistilBERT with a custom classification head.

    Architecture:
        DistilBERT encoder (6 transformer layers, hidden=768)
        → [CLS] token output (shape: 768)
        → Dropout(0.3)
        → Linear(768 → 256)
        → GELU activation
        → Dropout(0.3)
        → Linear(256 → num_classes)

    Why a 2-layer head?
        A single linear layer often underfits on emotion classification
        because emotional nuance lives in non-linear feature combinations.
        Two layers with GELU give the head enough capacity without
        risking overfitting on GoEmotions' ~58K samples.

    Args:
        base_model_name: HuggingFace model ID (default: distilbert-base-uncased)
        num_classes:     Number of emotion classes (default: 6)
        dropout:         Dropout rate for regularisation

    Returns:
        nn.Module — full model ready for fine-tuning or inference
    """
    import torch.nn as nn
    from transformers import DistilBertModel

    base_model_name = base_model_name or settings.distilbert_base_model
    num_classes = num_classes or settings.num_classes

    class DistilBertEmotionClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.distilbert = DistilBertModel.from_pretrained(base_model_name)
            hidden_size = self.distilbert.config.hidden_size  # 768

            # 2-layer classification head
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

        def forward(self, input_ids, attention_mask):
            # DistilBERT forward pass
            outputs = self.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # [CLS] token is the first token — represents full sentence
            cls_output = outputs.last_hidden_state[:, 0, :]  # (B, 768)
            logits = self.head(cls_output)                    # (B, num_classes)
            return logits

        def predict_proba(self, input_ids, attention_mask):
            import torch.nn.functional as F
            logits = self.forward(input_ids, attention_mask)
            return F.softmax(logits, dim=1)

    return DistilBertEmotionClassifier()


# ---------------------------------------------------------------
# Inference singleton
# ---------------------------------------------------------------

class TextEmotionClassifier:
    """
    Singleton inference engine for text-based emotion classification.

    Loads DistilBERT tokenizer + fine-tuned weights once and caches them.
    Supports optional INT8 dynamic quantization for ~1.5× speedup.

    Usage:
        clf = TextEmotionClassifier.get_instance()
        result = clf.predict("I don't understand this at all")
        print(result.emotion, result.confidence)
    """

    _instance: Optional["TextEmotionClassifier"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_path: str = None,
        use_quantization: bool = True,
    ):
        self.model_path = model_path or settings.distilbert_model_path
        self.use_quantization = use_quantization
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self.emotion_labels = settings.emotion_labels
        logger.info(f"TextEmotionClassifier init (path={self.model_path})")

    @classmethod
    def get_instance(
        cls,
        model_path: str = None,
        use_quantization: bool = True,
    ) -> "TextEmotionClassifier":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_path, use_quantization)
        return cls._instance

    def load_model(self) -> None:
        """
        Load tokenizer and fine-tuned model weights.
        Applies dynamic INT8 quantization if enabled and model exists.
        Falls back to base DistilBERT with random head if no fine-tuned
        weights found (useful during development before training).
        """
        if self._loaded:
            return

        import torch
        from transformers import DistilBertTokenizerFast

        model_path = Path(self.model_path)
        logger.info(f"Loading TextEmotionClassifier from {model_path}...")
        t0 = time.time()

        # Load tokenizer
        if (model_path / "tokenizer_config.json").exists():
            # Load fine-tuned tokenizer
            self._tokenizer = DistilBertTokenizerFast.from_pretrained(
                str(model_path)
            )
        else:
            # Fall back to base tokenizer
            logger.warning(
                "Fine-tuned tokenizer not found. "
                "Using base distilbert-base-uncased tokenizer. "
                "Run training/finetune_distilbert.py to fine-tune."
            )
            self._tokenizer = DistilBertTokenizerFast.from_pretrained(
                settings.distilbert_base_model
            )

        # Load model weights
        weights_path = model_path / "best_model.pt"
        if weights_path.exists():
            self._model = build_distilbert_classifier()
            checkpoint = torch.load(str(weights_path), map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self._model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self._model.load_state_dict(checkpoint)
            logger.info("Loaded fine-tuned DistilBERT weights.")
        else:
            logger.warning(
                f"Fine-tuned weights not found at {weights_path}. "
                "Using base DistilBERT + random head (untrained). "
                "Predictions will be random until you run finetune_distilbert.py."
            )
            self._model = build_distilbert_classifier()

        self._model.eval()

        # Apply dynamic quantization (INT8) for CPU speedup
        if self.use_quantization:
            self._model = self._apply_quantization(self._model)

        elapsed = (time.time() - t0) * 1000
        logger.info(f"TextEmotionClassifier loaded in {elapsed:.0f}ms")
        self._loaded = True

    @staticmethod
    def _apply_quantization(model) -> object:
        """
        Apply PyTorch dynamic quantization to Linear layers.

        Dynamic quantization:
        - Weights stored as INT8 (not FP32) — 4× memory reduction
        - Activations quantized dynamically at inference time
        - No calibration dataset needed (unlike static quantization)
        - Typically 1.3-1.8× speedup on CPU for transformer models
        - Negligible accuracy loss (<0.5% on classification tasks)
        """
        import torch
        logger.info("Applying INT8 dynamic quantization to Linear layers...")
        try:
            quantized = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},  # Only quantize Linear layers
                dtype=torch.qint8,
            )
            logger.info("Quantization applied successfully.")
            return quantized
        except Exception as e:
            logger.warning(f"Quantization failed ({e}), using FP32 model.")
            return model

    def predict(self, text: str) -> TextEmotionResult:
        """
        Classify emotion from a text string.

        Args:
            text: Any text string (student speech transcript)

        Returns:
            TextEmotionResult with emotion, confidence, probabilities
        """
        self.load_model()

        import torch

        if not text or not text.strip():
            # Return neutral for empty input
            return TextEmotionResult(
                text=text,
                emotion="neutral",
                confidence=1.0,
                probabilities={l: (1.0 if l == "neutral" else 0.0)
                               for l in self.emotion_labels},
                latency_ms=0.0,
                token_count=0,
            )

        t0 = time.time()
        text_clean = text.strip()

        # Tokenise
        encoding = self._tokenizer(
            text_clean,
            max_length=settings.distilbert_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_count = int(attention_mask.sum().item())

        # Inference
        with torch.no_grad():
            probs = self._model.predict_proba(input_ids, attention_mask)

        probs_np = probs.squeeze(0).numpy()
        pred_idx = int(probs_np.argmax())

        latency_ms = (time.time() - t0) * 1000

        result = TextEmotionResult(
            text=text_clean,
            emotion=self.emotion_labels[pred_idx],
            confidence=float(probs_np[pred_idx]),
            probabilities={
                label: float(probs_np[i])
                for i, label in enumerate(self.emotion_labels)
            },
            latency_ms=latency_ms,
            token_count=token_count,
        )

        logger.debug(
            f"Text emotion: '{text_clean[:50]}' → "
            f"{result.emotion} ({result.confidence:.1%}) in {latency_ms:.0f}ms"
        )
        return result

    def predict_batch(self, texts: List[str]) -> List[TextEmotionResult]:
        """
        Classify emotions for a batch of texts (more efficient than one-by-one).
        Used during evaluation of the fine-tuned model.
        """
        self.load_model()
        import torch

        t0 = time.time()
        encodings = self._tokenizer(
            texts,
            max_length=settings.distilbert_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            probs = self._model.predict_proba(
                encodings["input_ids"],
                encodings["attention_mask"],
            )

        probs_np = probs.numpy()
        total_ms = (time.time() - t0) * 1000
        per_ms = total_ms / len(texts)

        results = []
        for i, text in enumerate(texts):
            pred_idx = int(probs_np[i].argmax())
            results.append(TextEmotionResult(
                text=text,
                emotion=self.emotion_labels[pred_idx],
                confidence=float(probs_np[i][pred_idx]),
                probabilities={
                    label: float(probs_np[i][j])
                    for j, label in enumerate(self.emotion_labels)
                },
                latency_ms=per_ms,
                token_count=0,
            ))
        return results

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------

def predict_text_emotion(
    text: str,
    model_path: str = None,
) -> TextEmotionResult:
    """
    One-shot text emotion prediction using the global singleton.

    Args:
        text:       Input text (student transcript)
        model_path: Override default model path

    Returns:
        TextEmotionResult
    """
    clf = TextEmotionClassifier.get_instance(model_path)
    return clf.predict(text)
