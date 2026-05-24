# ============================================================
# modules/fusion.py
# Multimodal Fusion Engine
#
# Strategy: Weighted Late Fusion
#
#   final_probs = α × audio_probs + β × text_probs
#
# Default weights: α=0.6 (audio), β=0.4 (text)
#
# Why these weights?
#   - Audio prosody (pitch, energy, speaking rate) is a more
#     direct physiological signal of emotion than word choice
#   - Students often use polite/neutral words even when frustrated
#     (social desirability bias in text)
#   - Audio signal degrades with noise → dynamic adjustment pulls
#     weight toward text when audio confidence is low
#
# Beyond simple weighted average, this module implements:
#   1. Confidence-aware dynamic weight adjustment
#   2. Agreement bonus (boost confidence when both models agree)
#   3. Conflict detection and resolution
#   4. Temporal smoothing across multiple predictions
#   5. Full audit trail (every decision is explainable)
# ============================================================

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np

from config.settings import settings
from modules.logger import logger


# ---------------------------------------------------------------
# Input / Output types
# ---------------------------------------------------------------

@dataclass
class FusionInput:
    """
    Holds both modality predictions before fusion.
    Constructed from AudioEmotionResult + TextEmotionResult.
    """
    # Audio model outputs
    audio_probs: Dict[str, float]      # {emotion: probability}
    audio_emotion: str                 # top prediction
    audio_confidence: float            # max probability

    # Text model outputs
    text_probs: Dict[str, float]
    text_emotion: str
    text_confidence: float

    # Optional metadata
    transcript: str = ""
    audio_latency_ms: float = 0.0
    text_latency_ms: float = 0.0


@dataclass
class FusionResult:
    """
    Full output from the fusion engine.

    Attributes:
        emotion:         Final predicted emotion label
        confidence:      Confidence in final prediction [0, 1]
        probabilities:   Fused probability distribution
        audio_emotion:   What the audio model predicted
        text_emotion:    What the text model predicted
        models_agree:    True if both models predicted the same emotion
        fusion_weights:  Actual α and β used (may differ from defaults
                         if dynamic adjustment was applied)
        conflict_resolved: True if models disagreed and audio won
        latency_ms:      Total fusion computation time
        explanation:     Human-readable string explaining the decision
    """
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    audio_emotion: str
    text_emotion: str
    models_agree: bool
    fusion_weights: Tuple[float, float]     # (audio_weight, text_weight)
    conflict_resolved: bool = False
    latency_ms: float = 0.0
    explanation: str = ""

    def to_dict(self) -> Dict:
        return {
            "emotion": self.emotion,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "audio_emotion": self.audio_emotion,
            "text_emotion": self.text_emotion,
            "models_agree": self.models_agree,
            "fusion_weights": {
                "audio": round(self.fusion_weights[0], 3),
                "text": round(self.fusion_weights[1], 3),
            },
            "conflict_resolved": self.conflict_resolved,
            "latency_ms": round(self.latency_ms, 1),
            "explanation": self.explanation,
        }

    @property
    def is_uncertain(self) -> bool:
        """True when fused confidence is below threshold."""
        return self.confidence < 0.40


# ---------------------------------------------------------------
# Core fusion logic
# ---------------------------------------------------------------

class MultimodalFusion:
    """
    Late fusion engine combining audio and text emotion predictions.

    Fusion pipeline:
        1. Validate + normalise input probability vectors
        2. Compute dynamic weights based on per-modality confidence
        3. Compute weighted average of probability distributions
        4. Apply agreement bonus if both models predict same emotion
        5. Select argmax as final prediction
        6. Generate human-readable explanation
        7. Update temporal smoothing buffer

    Usage:
        fusion = MultimodalFusion()
        result = fusion.fuse(audio_result, text_result)
    """

    # Emotion adjacency — used in conflict resolution.
    # Emotions that are "close" to each other; resolving between
    # them uses the higher-confidence model directly.
    ADJACENT_EMOTIONS = {
        "confused":   {"frustrated", "neutral"},
        "frustrated": {"confused", "bored"},
        "bored":      {"neutral", "frustrated"},
        "happy":      {"engaged", "neutral"},
        "engaged":    {"happy", "neutral"},
        "neutral":    {"happy", "bored", "confused"},
    }

    def __init__(
        self,
        audio_weight: float = None,
        text_weight: float = None,
        use_dynamic_weights: bool = True,
        use_agreement_bonus: bool = True,
        smoothing_window: int = 3,
    ):
        """
        Args:
            audio_weight:         Default audio modality weight (α)
            text_weight:          Default text modality weight (β)
            use_dynamic_weights:  Adjust weights based on confidence
            use_agreement_bonus:  Boost confidence when models agree
            smoothing_window:     Rolling window for temporal smoothing
                                  (0 to disable)
        """
        raw_audio = audio_weight or settings.audio_fusion_weight
        raw_text = text_weight or settings.text_fusion_weight

        # Normalise weights so they always sum to 1
        total = raw_audio + raw_text
        self.default_audio_w = raw_audio / total
        self.default_text_w = raw_text / total

        self.use_dynamic_weights = use_dynamic_weights
        self.use_agreement_bonus = use_agreement_bonus
        self.smoothing_window = smoothing_window

        # Temporal smoothing buffer: stores last N probability dicts
        self._smooth_buffer: deque = deque(maxlen=smoothing_window)

        logger.info(
            f"MultimodalFusion init: "
            f"audio_w={self.default_audio_w:.2f}, "
            f"text_w={self.default_text_w:.2f}, "
            f"dynamic={use_dynamic_weights}, "
            f"smoothing={smoothing_window}"
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def fuse(
        self,
        audio_input,   # AudioEmotionResult or dict
        text_input,    # TextEmotionResult or dict
        transcript: str = "",
    ) -> FusionResult:
        """
        Fuse audio and text emotion predictions.

        Accepts both result dataclass objects (from modules) and plain dicts
        (for API use where results may be serialised/deserialised).

        Args:
            audio_input:  AudioEmotionResult or dict with 'probabilities' key
            text_input:   TextEmotionResult or dict with 'probabilities' key
            transcript:   Optional transcript text for logging

        Returns:
            FusionResult with final emotion, confidence, and full audit trail
        """
        t0 = time.time()

        # ---- 1. Extract and normalise probability vectors ----
        audio_probs = self._extract_probs(audio_input)
        text_probs = self._extract_probs(text_input)

        audio_probs = self._normalise_probs(audio_probs)
        text_probs = self._normalise_probs(text_probs)

        audio_conf = max(audio_probs.values())
        text_conf = max(text_probs.values())
        audio_emotion = max(audio_probs, key=audio_probs.get)
        text_emotion = max(text_probs, key=text_probs.get)

        fi = FusionInput(
            audio_probs=audio_probs,
            audio_emotion=audio_emotion,
            audio_confidence=audio_conf,
            text_probs=text_probs,
            text_emotion=text_emotion,
            text_confidence=text_conf,
            transcript=transcript,
        )

        # ---- 2. Compute dynamic weights ----
        audio_w, text_w = self._compute_weights(fi)

        # ---- 3. Weighted average fusion ----
        fused_probs = self._weighted_average(audio_probs, text_probs, audio_w, text_w)

        # ---- 4. Agreement bonus ----
        models_agree = (audio_emotion == text_emotion)
        if self.use_agreement_bonus and models_agree:
            fused_probs = self._apply_agreement_bonus(fused_probs, audio_emotion)

        # ---- 5. Conflict resolution ----
        conflict_resolved = False
        if not models_agree:
            fused_probs, conflict_resolved = self._resolve_conflict(
                fused_probs, fi, audio_w, text_w
            )

        # ---- 6. Temporal smoothing ----
        if self.smoothing_window > 0:
            fused_probs = self._temporal_smooth(fused_probs)

        # ---- 7. Final prediction ----
        final_emotion = max(fused_probs, key=fused_probs.get)
        final_confidence = fused_probs[final_emotion]

        # ---- 8. Generate explanation ----
        explanation = self._explain(
            fi, final_emotion, final_confidence,
            audio_w, text_w, models_agree, conflict_resolved
        )

        latency_ms = (time.time() - t0) * 1000

        result = FusionResult(
            emotion=final_emotion,
            confidence=final_confidence,
            probabilities=fused_probs,
            audio_emotion=audio_emotion,
            text_emotion=text_emotion,
            models_agree=models_agree,
            fusion_weights=(audio_w, text_w),
            conflict_resolved=conflict_resolved,
            latency_ms=latency_ms,
            explanation=explanation,
        )

        logger.debug(
            f"Fusion: audio={audio_emotion}({audio_conf:.0%}) + "
            f"text={text_emotion}({text_conf:.0%}) "
            f"→ {final_emotion}({final_confidence:.0%}) "
            f"[agree={models_agree}] [{latency_ms:.1f}ms]"
        )

        return result

    # ------------------------------------------------------------------
    # Step 2: Dynamic weight computation
    # ------------------------------------------------------------------

    def _compute_weights(self, fi: FusionInput) -> Tuple[float, float]:
        """
        Adjust fusion weights based on per-modality confidence.

        Rules:
          - If audio confidence is very low (<0.35): shift 20% weight to text
            (audio model is uncertain, likely noisy input)
          - If text confidence is very low (<0.35): shift 20% weight to audio
            (text is very short/ambiguous — "okay", "hmm")
          - If both are low: fall back to default weights
          - Otherwise: use default weights

        This makes the system robust to:
          - Background noise (degrades audio confidence)
          - Very short transcripts like "yeah" or "uh huh"
        """
        if not self.use_dynamic_weights:
            return self.default_audio_w, self.default_text_w

        audio_w = self.default_audio_w
        text_w = self.default_text_w

        LOW_CONF_THRESHOLD = 0.35
        SHIFT_AMOUNT = 0.20

        audio_low = fi.audio_confidence < LOW_CONF_THRESHOLD
        text_low = fi.text_confidence < LOW_CONF_THRESHOLD

        if audio_low and not text_low:
            # Audio uncertain → lean on text
            audio_w = max(0.10, audio_w - SHIFT_AMOUNT)
            text_w = 1.0 - audio_w
            logger.debug(
                f"Dynamic weights: audio low ({fi.audio_confidence:.0%}) "
                f"→ audio_w={audio_w:.2f}, text_w={text_w:.2f}"
            )

        elif text_low and not audio_low:
            # Text uncertain → lean on audio
            text_w = max(0.10, text_w - SHIFT_AMOUNT)
            audio_w = 1.0 - text_w
            logger.debug(
                f"Dynamic weights: text low ({fi.text_confidence:.0%}) "
                f"→ audio_w={audio_w:.2f}, text_w={text_w:.2f}"
            )

        return audio_w, text_w

    # ------------------------------------------------------------------
    # Step 3: Weighted average
    # ------------------------------------------------------------------

    def _weighted_average(
        self,
        audio_probs: Dict[str, float],
        text_probs: Dict[str, float],
        audio_w: float,
        text_w: float,
    ) -> Dict[str, float]:
        """
        Compute element-wise weighted average of two probability dicts.

        formula:
            fused[emotion] = audio_w × audio_probs[emotion]
                           + text_w  × text_probs[emotion]
        """
        fused = {}
        for label in settings.emotion_labels:
            a = audio_probs.get(label, 0.0)
            t = text_probs.get(label, 0.0)
            fused[label] = audio_w * a + text_w * t

        return self._normalise_probs(fused)

    # ------------------------------------------------------------------
    # Step 4: Agreement bonus
    # ------------------------------------------------------------------

    def _apply_agreement_bonus(
        self,
        fused_probs: Dict[str, float],
        agreed_emotion: str,
        bonus: float = 0.08,
    ) -> Dict[str, float]:
        """
        When both models predict the same emotion, boost its probability.

        Rationale: Agreement across modalities is strong evidence.
        If audio says "frustrated" AND text says "frustrated", the
        system should be more confident than the weighted average alone.

        The bonus is taken from all other classes proportionally so
        probabilities still sum to 1.
        """
        boosted = dict(fused_probs)
        current = boosted[agreed_emotion]
        available_bonus = min(bonus, 1.0 - current)  # cap at 1.0

        if available_bonus <= 0:
            return boosted

        # Add bonus to agreed class
        boosted[agreed_emotion] = current + available_bonus

        # Subtract proportionally from other classes
        other_total = sum(v for k, v in boosted.items() if k != agreed_emotion)
        if other_total > 0:
            for label in boosted:
                if label != agreed_emotion:
                    boosted[label] -= available_bonus * (boosted[label] / other_total)

        return self._normalise_probs(boosted)

    # ------------------------------------------------------------------
    # Step 5: Conflict resolution
    # ------------------------------------------------------------------

    def _resolve_conflict(
        self,
        fused_probs: Dict[str, float],
        fi: FusionInput,
        audio_w: float,
        text_w: float,
    ) -> Tuple[Dict[str, float], bool]:
        """
        Handle cases where audio and text models strongly disagree.

        A conflict is "strong" when:
          - The two predicted emotions are NOT adjacent (semantically distant)
          - AND at least one model has confidence > 0.60

        Resolution strategy:
          - Identify which model is more confident
          - Shift 15% additional weight toward the more confident model
          - Re-compute weighted average with adjusted weights

        If emotions ARE adjacent (e.g., confused vs frustrated), the
        simple weighted average is usually fine — no additional action needed.

        Returns:
            (updated_probs, conflict_was_resolved_bool)
        """
        audio_emotion = fi.audio_emotion
        text_emotion = fi.text_emotion

        # Check if this is a "strong" conflict
        adjacent = self.ADJACENT_EMOTIONS.get(audio_emotion, set())
        is_adjacent = text_emotion in adjacent

        strong_conflict = (
            not is_adjacent
            and max(fi.audio_confidence, fi.text_confidence) > 0.60
        )

        if not strong_conflict:
            return fused_probs, False

        # Resolve by boosting the more confident modality
        RESOLUTION_BOOST = 0.15
        if fi.audio_confidence >= fi.text_confidence:
            new_audio_w = min(0.95, audio_w + RESOLUTION_BOOST)
            new_text_w = 1.0 - new_audio_w
            winner = "audio"
        else:
            new_text_w = min(0.95, text_w + RESOLUTION_BOOST)
            new_audio_w = 1.0 - new_text_w
            winner = "text"

        resolved_probs = self._weighted_average(
            fi.audio_probs, fi.text_probs, new_audio_w, new_text_w
        )

        logger.debug(
            f"Conflict resolution: {audio_emotion} vs {text_emotion} "
            f"(non-adjacent, strong) → {winner} wins "
            f"(audio_w={new_audio_w:.2f}, text_w={new_text_w:.2f})"
        )

        return resolved_probs, True

    # ------------------------------------------------------------------
    # Step 6: Temporal smoothing
    # ------------------------------------------------------------------

    def _temporal_smooth(
        self,
        current_probs: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Smooth predictions over the last N frames (rolling average).

        Why this matters for learning systems:
          Emotion is not a static snapshot — a student who is "engaged"
          for 5 minutes and then says one confused sentence is probably
          still engaged overall. Smoothing prevents over-reaction to
          transient predictions.

        Window size 3 means:
          smoothed = mean([t-2_probs, t-1_probs, current_probs])

        The buffer auto-fills with the current prediction on first call.
        """
        self._smooth_buffer.append(current_probs)

        # Average all buffered probability dicts
        smoothed = {label: 0.0 for label in settings.emotion_labels}
        n = len(self._smooth_buffer)

        for past_probs in self._smooth_buffer:
            for label in settings.emotion_labels:
                smoothed[label] += past_probs.get(label, 0.0) / n

        return self._normalise_probs(smoothed)

    def reset_smoothing(self) -> None:
        """Clear the temporal smoothing buffer (call at session start)."""
        self._smooth_buffer.clear()
        logger.debug("Temporal smoothing buffer cleared.")

    # ------------------------------------------------------------------
    # Step 8: Explanation generation
    # ------------------------------------------------------------------

    def _explain(
        self,
        fi: FusionInput,
        final_emotion: str,
        final_confidence: float,
        audio_w: float,
        text_w: float,
        models_agree: bool,
        conflict_resolved: bool,
    ) -> str:
        """
        Generate a human-readable explanation of the fusion decision.
        Used in the dashboard and research report.
        """
        parts = []

        parts.append(
            f"Audio model: {fi.audio_emotion} ({fi.audio_confidence:.0%}) | "
            f"Text model: {fi.text_emotion} ({fi.text_confidence:.0%})"
        )

        if models_agree:
            parts.append(
                f"Both models agree on '{final_emotion}' → agreement bonus applied."
            )
        elif conflict_resolved:
            winner = "audio" if fi.audio_confidence >= fi.text_confidence else "text"
            parts.append(
                f"Conflict between '{fi.audio_emotion}' and '{fi.text_emotion}' "
                f"(non-adjacent) → {winner} model wins (higher confidence)."
            )
        else:
            parts.append(
                f"Adjacent disagreement ('{fi.audio_emotion}' vs '{fi.text_emotion}') "
                f"→ weighted average used."
            )

        if audio_w != self.default_audio_w:
            parts.append(
                f"Dynamic weight adjustment: "
                f"audio_w={audio_w:.2f}, text_w={text_w:.2f} "
                f"(default: {self.default_audio_w:.2f}/{self.default_text_w:.2f})"
            )
        else:
            parts.append(f"Weights: audio={audio_w:.2f}, text={text_w:.2f} (default).")

        parts.append(
            f"Final prediction: '{final_emotion}' at {final_confidence:.0%} confidence."
        )

        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_probs(result) -> Dict[str, float]:
        """
        Extract probability dict from either a result dataclass or a plain dict.
        Handles both AudioEmotionResult and TextEmotionResult objects.
        """
        if isinstance(result, dict):
            return result.get("probabilities", {})
        elif hasattr(result, "probabilities"):
            return result.probabilities
        else:
            raise TypeError(
                f"Expected result dataclass or dict with 'probabilities' key, "
                f"got {type(result)}"
            )

    @staticmethod
    def _normalise_probs(probs: Dict[str, float]) -> Dict[str, float]:
        """
        Ensure probabilities sum to exactly 1.0.
        Clips negatives and renormalises after bonus/penalty operations.
        """
        # Clip negatives (can occur after bonus subtraction)
        clipped = {k: max(0.0, v) for k, v in probs.items()}
        total = sum(clipped.values())
        if total <= 0:
            # Fallback: uniform distribution
            n = len(settings.emotion_labels)
            return {k: 1.0 / n for k in settings.emotion_labels}
        return {k: v / total for k, v in clipped.items()}

    @staticmethod
    def build_uniform_probs() -> Dict[str, float]:
        """Return uniform probability distribution (used as fallback)."""
        n = len(settings.emotion_labels)
        return {label: 1.0 / n for label in settings.emotion_labels}

    @staticmethod
    def build_single_class_probs(emotion: str, confidence: float = 0.9) -> Dict[str, float]:
        """
        Build a mock probability dict where one class dominates.
        Useful for testing and for the dummy/fallback case.
        """
        remaining = 1.0 - confidence
        other_count = len(settings.emotion_labels) - 1
        probs = {
            label: (confidence if label == emotion else remaining / other_count)
            for label in settings.emotion_labels
        }
        return probs


# ---------------------------------------------------------------
# Session-level fusion manager
# ---------------------------------------------------------------

class FusionSession:
    """
    Manages fusion state across a full tutoring session.

    Tracks:
      - Emotion history (timeline for dashboard)
      - Running emotion counts (for teacher analytics)
      - Dominant emotion over the session
      - Session-level confidence statistics

    Usage:
        session = FusionSession()
        for each_utterance:
            result = session.process(audio_result, text_result, transcript)
        summary = session.get_summary()
    """

    def __init__(
        self,
        audio_weight: float = None,
        text_weight: float = None,
    ):
        self.fusion = MultimodalFusion(
            audio_weight=audio_weight,
            text_weight=text_weight,
        )
        self._history: List[Dict] = []
        self._start_time = time.time()

    def process(
        self,
        audio_result,
        text_result,
        transcript: str = "",
    ) -> FusionResult:
        """
        Process one utterance and record to history.

        Returns FusionResult with complete fusion decision.
        """
        result = self.fusion.fuse(audio_result, text_result, transcript)

        self._history.append({
            "timestamp": round(time.time() - self._start_time, 1),
            "emotion": result.emotion,
            "confidence": round(result.confidence, 3),
            "audio_emotion": result.audio_emotion,
            "text_emotion": result.text_emotion,
            "models_agree": result.models_agree,
            "transcript": transcript[:100] if transcript else "",
        })

        return result

    def get_history(self) -> List[Dict]:
        """Return the full emotion timeline."""
        return list(self._history)

    def get_summary(self) -> Dict:
        """
        Compute session-level statistics.

        Returns:
            Dict with dominant_emotion, emotion_counts, avg_confidence,
            agreement_rate, session_duration_s, n_utterances
        """
        if not self._history:
            return {"n_utterances": 0}

        from collections import Counter

        emotions = [h["emotion"] for h in self._history]
        emotion_counts = dict(Counter(emotions))
        dominant = max(emotion_counts, key=emotion_counts.get)

        confidences = [h["confidence"] for h in self._history]
        agreements = [h["models_agree"] for h in self._history]

        return {
            "dominant_emotion": dominant,
            "emotion_counts": emotion_counts,
            "emotion_percentages": {
                e: round(c / len(emotions) * 100, 1)
                for e, c in emotion_counts.items()
            },
            "avg_confidence": round(float(np.mean(confidences)), 3),
            "agreement_rate": round(float(np.mean(agreements)), 3),
            "session_duration_s": round(time.time() - self._start_time, 1),
            "n_utterances": len(self._history),
        }

    def reset(self) -> None:
        """Reset session state (new student or new topic)."""
        self._history.clear()
        self.fusion.reset_smoothing()
        self._start_time = time.time()
        logger.info("FusionSession reset.")


# ---------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------

_default_fusion = None

def fuse_emotions(
    audio_result,
    text_result,
    transcript: str = "",
) -> FusionResult:
    """
    One-shot fusion using the default MultimodalFusion instance.

    Args:
        audio_result: AudioEmotionResult or dict
        text_result:  TextEmotionResult or dict
        transcript:   Optional transcript text

    Returns:
        FusionResult
    """
    global _default_fusion
    if _default_fusion is None:
        _default_fusion = MultimodalFusion()
    return _default_fusion.fuse(audio_result, text_result, transcript)
