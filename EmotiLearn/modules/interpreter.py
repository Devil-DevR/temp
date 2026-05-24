# ============================================================
# modules/interpreter.py
# Emotion Interpretation Engine
#
# Converts a FusionResult into a structured LearningStrategy.
# This is the "pedagogical brain" of the system — it translates
# raw emotion signals into actionable teaching decisions.
#
# Design:
#   - Rule-based core (fast, deterministic, explainable)
#   - Hysteresis: prevents strategy flip-flopping on ambiguous emotions
#   - Difficulty tracker: maintains current level and adjusts it
#   - Session context: considers recent history before deciding
# ============================================================

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

from config.settings import settings, EMOTION_STRATEGIES
from modules.logger import logger


# ---------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------

@dataclass
class LearningStrategy:
    """
    A fully resolved teaching decision derived from an emotion.

    Attributes:
        emotion:          The detected emotion that triggered this strategy
        strategy_name:    Internal strategy identifier (e.g. 'simplify_and_reteach')
        message:          Opening message to show the student
        activity:         What activity to generate (e.g. 'interactive_quiz')
        difficulty_level: Current absolute difficulty level [1–10]
        difficulty_delta: How much difficulty changed this turn
        confidence:       Emotion confidence that triggered this decision
        emoji:            Emoji for the dashboard emotion display
        explanation:      Why this strategy was chosen
        should_intervene: True when the system needs to act (not just observe)
    """
    emotion: str
    strategy_name: str
    message: str
    activity: str
    difficulty_level: int
    difficulty_delta: int
    confidence: float
    emoji: str
    explanation: str
    should_intervene: bool = True

    def to_dict(self) -> Dict:
        return {
            "emotion":          self.emotion,
            "strategy_name":    self.strategy_name,
            "message":          self.message,
            "activity":         self.activity,
            "difficulty_level": self.difficulty_level,
            "difficulty_delta": self.difficulty_delta,
            "confidence":       round(self.confidence, 3),
            "emoji":            self.emoji,
            "explanation":      self.explanation,
            "should_intervene": self.should_intervene,
        }


@dataclass
class SessionState:
    """
    Tracks the evolving state of a student's learning session.

    Maintained across turns so the interpreter can make
    context-aware decisions (e.g. don't simplify if we just
    simplified two turns ago and student is still confused —
    maybe try a different approach instead).
    """
    difficulty_level: int = 5          # Current difficulty [1–10]
    turn_count: int = 0
    consecutive_same_emotion: int = 0
    last_emotion: Optional[str] = None
    last_strategy: Optional[str] = None
    emotion_history: List[str] = field(default_factory=list)
    strategy_history: List[str] = field(default_factory=list)
    confused_streak: int = 0           # How many turns in a row the student was confused
    frustrated_streak: int = 0
    engaged_streak: int = 0

    def update(self, emotion: str, strategy: str) -> None:
        """Record a new turn."""
        self.turn_count += 1

        # Streak tracking
        if emotion == self.last_emotion:
            self.consecutive_same_emotion += 1
        else:
            self.consecutive_same_emotion = 0

        # Per-emotion streaks
        self.confused_streak    = self.confused_streak + 1    if emotion == "confused"    else 0
        self.frustrated_streak  = self.frustrated_streak + 1  if emotion == "frustrated"  else 0
        self.engaged_streak     = self.engaged_streak + 1     if emotion == "engaged"     else 0

        self.last_emotion = emotion
        self.last_strategy = strategy
        self.emotion_history.append(emotion)
        self.strategy_history.append(strategy)

    def clamp_difficulty(self, value: int) -> int:
        """Keep difficulty within [1, 10]."""
        return max(1, min(10, value))


# ---------------------------------------------------------------
# Interpretation engine
# ---------------------------------------------------------------

class EmotionInterpreter:
    """
    Maps a FusionResult (or emotion string) to a LearningStrategy.

    Core interpretation rules (beyond the base strategy map):

    1. HYSTERESIS — Require 2 consecutive low-confidence same-emotion
       predictions before changing strategy. Prevents single-frame
       noise from disrupting the lesson flow.

    2. STREAK ESCALATION — If a student has been confused for 3+
       consecutive turns, escalate from 'simplify' to 'alternative
       explanation' strategy. If frustrated for 2+ turns, add an
       encouragement prefix.

    3. DIFFICULTY BOUNDS — Never drop below level 1 or above 10.
       Never increase difficulty if last 2 emotions were negative.

    4. NEUTRAL HANDLING — Neutral is the "continue" signal. It does
       not trigger any adaptive change unless it has persisted for
       5+ turns (possible disengagement).

    Usage:
        interpreter = EmotionInterpreter()
        strategy = interpreter.interpret(fusion_result)
        print(strategy.message, strategy.activity)
    """

    # Minimum confidence to act on a prediction (else treat as neutral)
    CONFIDENCE_THRESHOLD = 0.38

    # How many turns of the same negative emotion before escalation
    CONFUSION_ESCALATION_TURNS  = 3
    FRUSTRATION_ESCALATION_TURNS = 2

    # Neutral persistence before flagging possible disengagement
    NEUTRAL_DISENGAGEMENT_TURNS = 5

    def __init__(self, initial_difficulty: int = 5):
        self.state = SessionState(difficulty_level=initial_difficulty)
        # Hysteresis buffer: store last 2 emotions for flip-flop prevention
        self._recent_emotions: deque = deque(maxlen=2)
        logger.info(
            f"EmotionInterpreter init (difficulty={initial_difficulty})"
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def interpret(self, fusion_result) -> LearningStrategy:
        """
        Convert a FusionResult into a LearningStrategy.

        Args:
            fusion_result: FusionResult object or dict with
                           'emotion' and 'confidence' keys

        Returns:
            LearningStrategy with full pedagogical decision
        """
        # Extract emotion + confidence from either object or dict
        if isinstance(fusion_result, dict):
            emotion    = fusion_result.get("emotion", "neutral")
            confidence = fusion_result.get("confidence", 0.5)
        else:
            emotion    = getattr(fusion_result, "emotion", "neutral")
            confidence = getattr(fusion_result, "confidence", 0.5)

        # Validate emotion label
        if emotion not in settings.emotion_labels:
            logger.warning(f"Unknown emotion '{emotion}', defaulting to neutral")
            emotion = "neutral"

        # Apply confidence threshold — uncertain predictions → neutral
        if confidence < self.CONFIDENCE_THRESHOLD:
            logger.debug(
                f"Confidence {confidence:.0%} below threshold "
                f"→ treating as neutral"
            )
            emotion = "neutral"
            confidence = confidence  # keep original for display

        # Apply hysteresis for strategy stability
        emotion = self._apply_hysteresis(emotion)

        # Resolve strategy (with streak escalation)
        strategy_key, message, activity, delta, explanation = \
            self._resolve_strategy(emotion, confidence)

        # Update difficulty
        new_difficulty = self.state.clamp_difficulty(
            self.state.difficulty_level + delta
        )
        self.state.difficulty_level = new_difficulty

        # Determine if system should actively intervene
        should_intervene = self._should_intervene(emotion, confidence)

        # Build strategy object
        base = EMOTION_STRATEGIES[emotion]
        strategy = LearningStrategy(
            emotion          = emotion,
            strategy_name    = strategy_key,
            message          = message,
            activity         = activity,
            difficulty_level = new_difficulty,
            difficulty_delta = delta,
            confidence       = confidence,
            emoji            = base["emoji"],
            explanation      = explanation,
            should_intervene = should_intervene,
        )

        # Update session state
        self.state.update(emotion, strategy_key)
        self._recent_emotions.append(emotion)

        logger.info(
            f"Strategy: {strategy_key} | emotion={emotion} "
            f"({confidence:.0%}) | difficulty={new_difficulty} "
            f"(Δ={delta:+d}) | intervene={should_intervene}"
        )

        return strategy

    # ------------------------------------------------------------------
    # Hysteresis
    # ------------------------------------------------------------------

    def _apply_hysteresis(self, emotion: str) -> str:
        """
        Prevent single-frame emotion flips from changing strategy.

        If the last 2 emotions were both X, a single Y prediction
        is probably noise — keep X unless Y appears twice in a row.

        Special case: if emotion is "confused" or "frustrated" (negative),
        we DO act on it immediately (one turn is enough).
        """
        ALWAYS_ACT_IMMEDIATELY = {"confused", "frustrated", "engaged"}

        if emotion in ALWAYS_ACT_IMMEDIATELY:
            return emotion  # Never delay on strong signals

        if len(self._recent_emotions) < 2:
            return emotion  # Not enough history yet

        prev1, prev2 = list(self._recent_emotions)[-1], list(self._recent_emotions)[-2]

        # If last 2 were the same and current differs → stick with last
        if prev1 == prev2 and emotion != prev1:
            logger.debug(
                f"Hysteresis: suppressing {emotion} → keeping {prev1}"
            )
            return prev1

        return emotion

    # ------------------------------------------------------------------
    # Strategy resolution with escalation logic
    # ------------------------------------------------------------------

    def _resolve_strategy(
        self,
        emotion: str,
        confidence: float,
    ) -> Tuple[str, str, str, int, str]:
        """
        Return (strategy_key, message, activity, difficulty_delta, explanation).

        Applies streak escalation and neutral disengagement detection
        on top of the base strategy map.
        """
        base = EMOTION_STRATEGIES[emotion]
        strategy_key = base["strategy"]
        message      = base["message_template"]
        activity     = base["activity"]
        delta        = base["difficulty_delta"]
        explanation  = f"Base strategy for '{emotion}'."

        # ---- Escalation: prolonged confusion ----
        if emotion == "confused" and \
                self.state.confused_streak >= self.CONFUSION_ESCALATION_TURNS:
            strategy_key = "alternative_explanation"
            message = (
                "We've tried the standard explanation a few times. "
                "Let me try a completely different approach with an analogy:"
            )
            activity    = "analogy_explanation"
            delta       = -1
            explanation = (
                f"Student has been confused for "
                f"{self.state.confused_streak} consecutive turns. "
                "Escalating to alternative explanation strategy."
            )
            logger.info(
                f"Confusion escalation triggered "
                f"(streak={self.state.confused_streak})"
            )

        # ---- Escalation: prolonged frustration ----
        elif emotion == "frustrated" and \
                self.state.frustrated_streak >= self.FRUSTRATION_ESCALATION_TURNS:
            strategy_key = "step_by_step_with_encouragement"
            message = (
                "I can see this is really challenging. That's completely normal — "
                "let's break it into the smallest possible steps and tackle them one at a time:"
            )
            activity    = "micro_steps"
            delta       = -1
            explanation = (
                f"Student has been frustrated for "
                f"{self.state.frustrated_streak} consecutive turns. "
                "Adding encouragement prefix."
            )

        # ---- Neutral disengagement detection ----
        elif emotion == "neutral" and \
                self.state.consecutive_same_emotion >= self.NEUTRAL_DISENGAGEMENT_TURNS:
            strategy_key = "re_engagement"
            message = (
                "Let's mix things up a bit! Here's something a little different:"
            )
            activity    = "interactive_quiz"
            delta       = 0
            explanation = (
                f"Student has been neutral for "
                f"{self.state.consecutive_same_emotion} turns — "
                "possible disengagement, switching to re-engagement strategy."
            )
            logger.info("Neutral disengagement detected → re-engagement")

        # ---- Do not increase difficulty after recent negative emotions ----
        elif delta > 0 and self._recent_negative_history():
            delta       = 0
            explanation = (
                f"Difficulty increase suppressed: recent negative emotion history. "
                f"Holding at level {self.state.difficulty_level}."
            )

        return strategy_key, message, activity, delta, explanation

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_intervene(self, emotion: str, confidence: float) -> bool:
        """
        Decide whether the tutor should actively intervene or just
        silently observe this turn.

        No intervention when:
          - emotion is neutral AND we haven't detected disengagement
          - confidence is moderate and emotion just changed (could be noise)
        """
        PASSIVE_EMOTIONS = {"neutral"}
        if emotion in PASSIVE_EMOTIONS and \
                self.state.consecutive_same_emotion < self.NEUTRAL_DISENGAGEMENT_TURNS:
            return False
        return True

    def _recent_negative_history(self, window: int = 2) -> bool:
        """Return True if any of the last `window` emotions were negative."""
        NEGATIVE = {"confused", "frustrated", "bored"}
        recent = list(self.state.emotion_history)[-window:]
        return any(e in NEGATIVE for e in recent)

    def reset(self, difficulty: int = 5) -> None:
        """Reset session state for a new student or new topic."""
        self.state = SessionState(difficulty_level=difficulty)
        self._recent_emotions.clear()
        logger.info(f"EmotionInterpreter reset (difficulty={difficulty})")

    @property
    def current_difficulty(self) -> int:
        return self.state.difficulty_level

    def get_session_snapshot(self) -> Dict:
        """Return a snapshot of current session state (for analytics)."""
        return {
            "turn_count":                self.state.turn_count,
            "current_difficulty":        self.state.difficulty_level,
            "last_emotion":              self.state.last_emotion,
            "last_strategy":             self.state.last_strategy,
            "confused_streak":           self.state.confused_streak,
            "frustrated_streak":         self.state.frustrated_streak,
            "engaged_streak":            self.state.engaged_streak,
            "recent_emotions":           list(self.state.emotion_history)[-5:],
        }


# ---------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------

def interpret_emotion(fusion_result) -> LearningStrategy:
    """
    One-shot interpretation using a fresh EmotionInterpreter.
    Useful for stateless single-turn API calls.
    """
    return EmotionInterpreter().interpret(fusion_result)
