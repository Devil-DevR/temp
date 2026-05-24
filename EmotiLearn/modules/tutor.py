# ============================================================
# modules/tutor.py
# Adaptive Tutor Response Generator
#
# Takes a LearningStrategy and generates a complete, context-aware
# tutor response: opening message + content + follow-up activity.
#
# Architecture:
#   - Template-based engine (fast, deterministic, no LLM required)
#   - Activity generators per emotion/strategy
#   - Topic-aware content bank (pluggable)
#   - Full TutorResponse dataclass for API/frontend consumption
#
# Why template-based instead of LLM?
#   - Latency: templates respond in <5ms vs ~2s for an LLM
#   - Reliability: no hallucination risk for educational content
#   - Predictability: teachers can audit and customise all responses
#   - Resource: runs on free-tier CPU with zero additional memory
#   LLM integration is provided as an optional upgrade path.
# ============================================================

import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from config.settings import settings, EMOTION_STRATEGIES
from modules.logger import logger


# ---------------------------------------------------------------
# Response data types
# ---------------------------------------------------------------

@dataclass
class ActivityItem:
    """A single learning activity (quiz question, exercise, etc.)."""
    activity_type: str           # 'quiz', 'fill_blank', 'analogy', 'steps', 'challenge'
    prompt: str                  # What to show the student
    options: List[str] = field(default_factory=list)   # MCQ options (if applicable)
    answer: Optional[str] = None                        # Correct answer (teacher view)
    hint: Optional[str] = None                          # Hint if student struggles

    def to_dict(self) -> Dict:
        return {
            "activity_type": self.activity_type,
            "prompt":        self.prompt,
            "options":       self.options,
            "answer":        self.answer,
            "hint":          self.hint,
        }


@dataclass
class TutorResponse:
    """
    Complete response from the Adaptive Tutor.

    Attributes:
        emotion:         Detected student emotion
        strategy_name:   Strategy that was applied
        opening_message: Empathetic opening sentence
        content:         Main instructional content
        activity:        Follow-up activity (quiz, steps, etc.)
        difficulty_level: Current difficulty level [1–10]
        emoji:           Emotion emoji for dashboard
        response_time_ms: How long generation took
        topic:           Current learning topic (if set)
    """
    emotion: str
    strategy_name: str
    opening_message: str
    content: str
    activity: Optional[ActivityItem]
    difficulty_level: int
    emoji: str
    response_time_ms: float = 0.0
    topic: str = "general"

    def to_dict(self) -> Dict:
        return {
            "emotion":          self.emotion,
            "strategy_name":    self.strategy_name,
            "opening_message":  self.opening_message,
            "content":          self.content,
            "activity":         self.activity.to_dict() if self.activity else None,
            "difficulty_level": self.difficulty_level,
            "emoji":            self.emoji,
            "response_time_ms": round(self.response_time_ms, 1),
            "topic":            self.topic,
        }

    @property
    def full_text(self) -> str:
        """Return opening + content as a single display string."""
        parts = [self.opening_message, "", self.content]
        if self.activity:
            parts += ["", f"📝 {self.activity.prompt}"]
            if self.activity.options:
                for i, opt in enumerate(self.activity.options, 1):
                    parts.append(f"   {i}. {opt}")
        return "\n".join(parts)


# ---------------------------------------------------------------
# Content bank
# ---------------------------------------------------------------

# Pluggable content bank: maps topic → difficulty → content blocks.
# In a real deployment, this would be loaded from a database or
# curriculum management system. Here we provide a rich default
# covering common STEM and general topics.

CONTENT_BANK: Dict[str, Dict[int, Dict[str, Any]]] = {
    "recursion": {
        1: {
            "simple":  "Think of recursion like Russian nesting dolls (Matryoshka). Each doll contains a smaller version of itself, until you reach the tiniest doll that doesn't open. That's the base case!",
            "analogy": "Imagine you're looking for your keys. You check each room. 'Are the keys here? No — check the next room.' You keep asking the same question in smaller spaces until you find them.",
            "steps":   ["Step 1: Identify the simplest case (base case) where the answer is known directly.",
                        "Step 2: Express the bigger problem in terms of a smaller, similar problem.",
                        "Step 3: Trust that the smaller problem will be solved the same way.",
                        "Step 4: Combine results as the recursion 'unwinds'."],
            "quiz":    [{"q": "What stops a recursive function from running forever?",
                         "options": ["A for loop", "The base case", "A variable", "A class"],
                         "answer": "The base case",
                         "hint": "Think about the smallest doll that can't be opened."}],
        },
        5: {
            "simple":  "Recursion is when a function calls itself with a smaller input until it reaches a base case. Classic examples: factorial(n) = n × factorial(n-1), and Fibonacci(n) = Fibonacci(n-1) + Fibonacci(n-2).",
            "analogy": "Think of a mirror facing another mirror — infinite reflections, but each one is slightly smaller. The base case is when the reflection becomes too small to see.",
            "steps":   ["Step 1: Write the base case (when to stop).",
                        "Step 2: Write the recursive case (call self with reduced input).",
                        "Step 3: Ensure each call moves toward the base case.",
                        "Step 4: Verify correctness with the call stack trace."],
            "quiz":    [{"q": "What is the time complexity of naive recursive Fibonacci?",
                         "options": ["O(n)", "O(n log n)", "O(2^n)", "O(n²)"],
                         "answer": "O(2^n)",
                         "hint": "Each call branches into two more calls."}],
        },
        9: {
            "simple":  "Advanced recursion includes tail recursion (last operation is the recursive call), memoization (caching subproblem results), and mutual recursion (A calls B, B calls A). Tail-recursive functions can be optimised to use O(1) stack space.",
            "analogy": "Memoized recursion is like a student who writes answers on sticky notes — instead of re-deriving every answer, they check their notes first.",
            "steps":   ["Step 1: Identify overlapping subproblems (where memoization helps).",
                        "Step 2: Add a cache dictionary to store computed results.",
                        "Step 3: Check cache before computing; store result after.",
                        "Step 4: Analyze if tail recursion optimization is possible."],
            "quiz":    [{"q": "What Python decorator enables automatic memoization?",
                         "options": ["@cache", "@functools.lru_cache", "@memo", "@lru_cache"],
                         "answer": "@functools.lru_cache",
                         "hint": "It's in the functools standard library module."}],
        },
    },
    "machine_learning": {
        1: {
            "simple":  "Machine learning lets computers learn patterns from examples, just like how you learned to recognise a dog from many pictures of dogs.",
            "analogy": "Imagine teaching a child to recognise apples. You show hundreds of apples of different colours and sizes. Eventually the child learns what makes an apple an apple, even without memorising each one.",
            "steps":   ["Step 1: Collect labelled examples (training data).",
                        "Step 2: Choose a model (the 'learner').",
                        "Step 3: Train the model on the examples.",
                        "Step 4: Test on new unseen examples."],
            "quiz":    [{"q": "What do we call the data used to train an ML model?",
                         "options": ["Test set", "Training set", "Validation set", "Feature set"],
                         "answer": "Training set",
                         "hint": "It's the data the model 'learns' from."}],
        },
        5: {
            "simple":  "Supervised learning uses labelled data (input → output pairs). The model learns a mapping f(x) → y by minimising a loss function. Common algorithms: linear regression, decision trees, SVMs, neural networks.",
            "analogy": "Training a model is like calibrating a scale. You place known weights and adjust the dial until readings are accurate. The 'dial adjustments' are gradient descent steps.",
            "steps":   ["Step 1: Define features (X) and target (y).",
                        "Step 2: Split into train/val/test sets.",
                        "Step 3: Choose and fit a model.",
                        "Step 4: Evaluate using appropriate metrics (accuracy, F1, etc.).",
                        "Step 5: Tune hyperparameters."],
            "quiz":    [{"q": "Which metric is best for imbalanced classification?",
                         "options": ["Accuracy", "F1-score", "MSE", "R-squared"],
                         "answer": "F1-score",
                         "hint": "This metric considers both false positives and false negatives."}],
        },
        9: {
            "simple":  "Advanced topics: gradient boosting (XGBoost, LightGBM), attention mechanisms (Transformers), contrastive learning (CLIP, SimCLR), and neural architecture search. Modern models combine self-supervised pre-training with task-specific fine-tuning.",
            "analogy": "Attention in Transformers is like a search engine query — the model 'queries' each token to decide how relevant it is to the current token being processed.",
            "steps":   ["Step 1: Understand the self-attention formula: Attention(Q,K,V) = softmax(QKᵀ/√d)V.",
                        "Step 2: Study multi-head attention and positional encoding.",
                        "Step 3: Implement a mini-transformer from scratch.",
                        "Step 4: Experiment with fine-tuning vs. full training."],
            "quiz":    [{"q": "What does the softmax(QKᵀ/√d) term compute?",
                         "options": ["Word embeddings", "Attention weights", "Loss values", "Gradient norms"],
                         "answer": "Attention weights",
                         "hint": "It tells the model how much to 'attend' to each token."}],
        },
    },
    "general": {
        1: {
            "simple":  "Let me explain this concept in the simplest way possible, using everyday language.",
            "analogy": "Think of it this way: imagine you're explaining it to someone who has never heard of it before.",
            "steps":   ["Step 1: Understand the basic definition.",
                        "Step 2: Find a familiar example from your daily life.",
                        "Step 3: Try to explain it back in your own words.",
                        "Step 4: Practice with a simple example."],
            "quiz":    [{"q": "Can you explain this concept back to me in one sentence?",
                         "options": [], "answer": None,
                         "hint": "Focus on the most important single idea."}],
        },
        5: {
            "simple":  "Now that you have the basics, let's explore this concept in more depth with formal definitions and worked examples.",
            "analogy": "Think about how this concept connects to things you already know well.",
            "steps":   ["Step 1: Review the formal definition.",
                        "Step 2: Work through a complete example step by step.",
                        "Step 3: Identify edge cases and exceptions.",
                        "Step 4: Apply the concept to a new problem."],
            "quiz":    [{"q": "Apply this concept to solve a medium-difficulty problem.",
                         "options": [], "answer": None,
                         "hint": "Break the problem into the steps we just discussed."}],
        },
        9: {
            "simple":  "At this advanced level, let's examine the theoretical underpinnings, edge cases, optimisations, and research-level extensions of this concept.",
            "analogy": "You've mastered the rules — now let's understand *why* the rules work and when to break them.",
            "steps":   ["Step 1: Analyse time and space complexity.",
                        "Step 2: Explore theoretical limits and proofs.",
                        "Step 3: Compare with alternative approaches.",
                        "Step 4: Design your own extension or optimisation."],
            "quiz":    [{"q": "Design and justify an optimisation or extension of this concept.",
                         "options": [], "answer": None,
                         "hint": "Consider real-world constraints and trade-offs."}],
        },
    },
}


def _get_content(topic: str, difficulty: int) -> Dict:
    """
    Look up the closest content block for a given topic + difficulty.
    Falls back to 'general' if topic not in content bank.
    Falls back to nearest difficulty level if exact match not found.
    """
    bank = CONTENT_BANK.get(topic, CONTENT_BANK["general"])
    levels = sorted(bank.keys())

    # Find nearest difficulty level
    closest = min(levels, key=lambda lvl: abs(lvl - difficulty))
    return bank[closest]


# ---------------------------------------------------------------
# Adaptive Tutor
# ---------------------------------------------------------------

class AdaptiveTutor:
    """
    Generates context-aware tutor responses from a LearningStrategy.

    Workflow:
        1. Select opening message (strategy-specific, empathetic)
        2. Retrieve content block for current topic + difficulty
        3. Select content type based on activity hint from strategy
        4. Generate follow-up activity
        5. Assemble TutorResponse

    Usage:
        tutor = AdaptiveTutor(topic="recursion")
        response = tutor.respond(strategy)
        print(response.full_text)
    """

    def __init__(self, topic: str = "general", seed: int = None):
        """
        Args:
            topic: Learning topic (must be a key in CONTENT_BANK or 'general')
            seed:  Random seed for reproducible responses
        """
        self.topic = topic if topic in CONTENT_BANK else "general"
        if seed is not None:
            random.seed(seed)
        logger.info(f"AdaptiveTutor init (topic={self.topic})")

    def respond(self, strategy) -> TutorResponse:
        """
        Generate a TutorResponse from a LearningStrategy.

        Args:
            strategy: LearningStrategy object or dict

        Returns:
            TutorResponse with full instructional content
        """
        t0 = time.time()

        # Normalise input
        if isinstance(strategy, dict):
            emotion          = strategy.get("emotion", "neutral")
            strategy_name    = strategy.get("strategy_name", "maintain_pace")
            opening_message  = strategy.get("message", "Let's continue.")
            activity_hint    = strategy.get("activity", "continue")
            difficulty_level = strategy.get("difficulty_level", 5)
            confidence       = strategy.get("confidence", 0.5)
            emoji            = strategy.get("emoji", "😐")
        else:
            emotion          = strategy.emotion
            strategy_name    = strategy.strategy_name
            opening_message  = strategy.message
            activity_hint    = strategy.activity
            difficulty_level = strategy.difficulty_level
            confidence       = strategy.confidence
            emoji            = strategy.emoji

        # Get content for this topic + difficulty
        content_block = _get_content(self.topic, difficulty_level)

        # Select content type based on activity hint
        content, activity = self._build_content_and_activity(
            emotion, activity_hint, difficulty_level, content_block
        )

        elapsed_ms = (time.time() - t0) * 1000

        response = TutorResponse(
            emotion          = emotion,
            strategy_name    = strategy_name,
            opening_message  = opening_message,
            content          = content,
            activity         = activity,
            difficulty_level = difficulty_level,
            emoji            = emoji,
            response_time_ms = elapsed_ms,
            topic            = self.topic,
        )

        logger.debug(
            f"Tutor response: {emotion} → {strategy_name} "
            f"| difficulty={difficulty_level} | {elapsed_ms:.1f}ms"
        )

        return response

    # ------------------------------------------------------------------
    # Content + activity builders
    # ------------------------------------------------------------------

    def _build_content_and_activity(
        self,
        emotion: str,
        activity_hint: str,
        difficulty: int,
        content_block: Dict,
    ):
        """Route to the appropriate content generator."""
        builders = {
            "simpler_explanation":   self._build_simpler,
            "analogy_explanation":   self._build_analogy,
            "step_by_step":          self._build_steps,
            "micro_steps":           self._build_micro_steps,
            "interactive_quiz":      self._build_quiz,
            "advance_topic":         self._build_advanced,
            "increase_difficulty":   self._build_advanced,
            "continue":              self._build_continue,
        }
        builder = builders.get(activity_hint, self._build_continue)
        return builder(difficulty, content_block)

    def _build_simpler(self, difficulty: int, cb: Dict):
        content  = cb.get("simple", "Let me explain this more simply.")
        activity = self._make_quiz_activity(cb, difficulty)
        return content, activity

    def _build_analogy(self, difficulty: int, cb: Dict):
        content  = cb.get("analogy", cb.get("simple", "Here's an analogy to help."))
        activity = ActivityItem(
            activity_type = "reflection",
            prompt        = "Does this analogy make sense to you? Try to extend it: what would the next step look like in the analogy?",
            hint          = "Think about what happens when you take the next step in the real-world scenario.",
        )
        return content, activity

    def _build_steps(self, difficulty: int, cb: Dict):
        steps   = cb.get("steps", ["Step 1: Understand the concept.", "Step 2: Apply it."])
        content = "\n".join(steps)
        activity = ActivityItem(
            activity_type = "checkpoint",
            prompt        = "After reading these steps, tell me: which step is the most unclear to you?",
            hint          = "It's completely fine if one step is confusing — we'll tackle it together.",
        )
        return content, activity

    def _build_micro_steps(self, difficulty: int, cb: Dict):
        steps = cb.get("steps", ["Step 1: ...", "Step 2: ...", "Step 3: ..."])
        # Break into even finer steps by splitting each step further
        micro = []
        for i, step in enumerate(steps, 1):
            micro.append(f"{step}")
            micro.append(f"   ✓ Pause here. Does step {i} make sense? (Yes/No)")
        content = "\n".join(micro)
        activity = ActivityItem(
            activity_type = "micro_checkpoint",
            prompt        = "Let's take it one micro-step at a time. Start with Step 1 only — what do you understand so far?",
            hint          = "Don't worry about the other steps yet. Focus only on step 1.",
        )
        return content, activity

    def _build_quiz(self, difficulty: int, cb: Dict):
        quizzes = cb.get("quiz", [])
        if quizzes:
            q    = random.choice(quizzes)
            act  = ActivityItem(
                activity_type = "multiple_choice",
                prompt        = q["q"],
                options       = q.get("options", []),
                answer        = q.get("answer"),
                hint          = q.get("hint"),
            )
        else:
            act = ActivityItem(
                activity_type = "open_question",
                prompt        = "Let's test your understanding: can you explain the core idea of this topic in your own words?",
                hint          = "Try to use an example from everyday life.",
            )
        content = "Let's make this more engaging with a quick challenge:"
        return content, act

    def _build_advanced(self, difficulty: int, cb: Dict):
        content  = cb.get("simple", "Here's the advanced version of this concept.")
        quizzes  = cb.get("quiz", [])
        if quizzes:
            q   = random.choice(quizzes)
            act = ActivityItem(
                activity_type = "challenge",
                prompt        = q["q"],
                options       = q.get("options", []),
                answer        = q.get("answer"),
                hint          = q.get("hint"),
            )
        else:
            act = ActivityItem(
                activity_type = "design_challenge",
                prompt        = f"Now that you've mastered the basics, design your own example or extension of this concept at difficulty level {difficulty}.",
                hint          = "Think about edge cases and what happens in extreme scenarios.",
            )
        return content, act

    def _build_continue(self, difficulty: int, cb: Dict):
        content = cb.get("simple", "Let's continue with the current material.")
        activity = ActivityItem(
            activity_type = "open_question",
            prompt        = "Before we move on, is there anything about what we've covered so far that you'd like to revisit?",
        )
        return content, activity

    def _make_quiz_activity(self, cb: Dict, difficulty: int) -> Optional[ActivityItem]:
        quizzes = cb.get("quiz", [])
        if not quizzes:
            return None
        q = random.choice(quizzes)
        return ActivityItem(
            activity_type = "multiple_choice",
            prompt        = q["q"],
            options       = q.get("options", []),
            answer        = q.get("answer"),
            hint          = q.get("hint"),
        )

    def set_topic(self, topic: str) -> None:
        """Change the learning topic mid-session."""
        self.topic = topic if topic in CONTENT_BANK else "general"
        logger.info(f"Tutor topic changed to: {self.topic}")


# ---------------------------------------------------------------
# Pipeline: full end-to-end response generation
# ---------------------------------------------------------------

class AdaptiveLearningPipeline:
    """
    Top-level pipeline that orchestrates the complete flow:
        FusionResult → Interpreter → Strategy → Tutor → TutorResponse

    This is the single object the API and Streamlit frontend
    interact with. It maintains full session state.

    Usage:
        pipeline = AdaptiveLearningPipeline(topic="machine_learning")
        response = pipeline.process(fusion_result, transcript="...")
        print(response.full_text)
        summary  = pipeline.get_session_summary()
    """

    def __init__(self, topic: str = "general", initial_difficulty: int = 5):
        from modules.interpreter import EmotionInterpreter

        self.interpreter = EmotionInterpreter(initial_difficulty=initial_difficulty)
        self.tutor       = AdaptiveTutor(topic=topic)
        self.topic       = topic
        self._responses: List[Dict] = []
        self._start_time = time.time()
        logger.info(
            f"AdaptiveLearningPipeline init "
            f"(topic={topic}, difficulty={initial_difficulty})"
        )

    def process(
        self,
        fusion_result,
        transcript: str = "",
    ) -> TutorResponse:
        """
        Full pipeline: FusionResult → Strategy → TutorResponse.

        Args:
            fusion_result: FusionResult object or dict
            transcript:    Optional transcript text (for logging)

        Returns:
            TutorResponse ready for display
        """
        # Step 1: Interpret emotion → strategy
        strategy = self.interpreter.interpret(fusion_result)

        # Step 2: Generate tutor response
        response = self.tutor.respond(strategy)

        # Step 3: Log for analytics
        self._responses.append({
            "timestamp":      round(time.time() - self._start_time, 1),
            "emotion":        strategy.emotion,
            "strategy":       strategy.strategy_name,
            "difficulty":     strategy.difficulty_level,
            "intervened":     strategy.should_intervene,
            "transcript":     transcript[:120] if transcript else "",
            "response_first_line": response.opening_message[:80],
        })

        return response

    def get_session_summary(self) -> Dict:
        """Return analytics summary of the full session."""
        from collections import Counter
        if not self._responses:
            return {"n_turns": 0}

        emotions   = [r["emotion"]   for r in self._responses]
        strategies = [r["strategy"]  for r in self._responses]
        difficulties = [r["difficulty"] for r in self._responses]

        return {
            "n_turns":           len(self._responses),
            "dominant_emotion":  Counter(emotions).most_common(1)[0][0],
            "emotion_counts":    dict(Counter(emotions)),
            "strategy_counts":   dict(Counter(strategies)),
            "difficulty_start":  difficulties[0] if difficulties else 5,
            "difficulty_end":    difficulties[-1] if difficulties else 5,
            "difficulty_trend":  difficulties,
            "intervention_rate": round(
                sum(r["intervened"] for r in self._responses) / len(self._responses), 2
            ),
            "session_duration_s": round(time.time() - self._start_time, 1),
            "topic":             self.topic,
        }

    def reset(self, topic: str = None, difficulty: int = 5) -> None:
        """Reset pipeline for a new student or topic."""
        self.interpreter.reset(difficulty=difficulty)
        if topic:
            self.topic = topic
            self.tutor.set_topic(topic)
        self._responses.clear()
        self._start_time = time.time()
        logger.info(f"AdaptiveLearningPipeline reset (topic={self.topic})")

    @property
    def current_difficulty(self) -> int:
        return self.interpreter.current_difficulty
