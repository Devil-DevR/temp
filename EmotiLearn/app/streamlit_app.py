#!/usr/bin/env python3
# ============================================================
# app/streamlit_app.py
# Emotion-Aware Learning Assistant — Streamlit Frontend
#
# Pages:
#   🎓 Student View  — real-time learning session with emotion-adaptive tutor
#   📊 Analytics     — teacher dashboard with emotion timeline + stats
#   ⚙️  Settings     — configure topic, difficulty, fusion weights
#
# Connects to FastAPI backend (api/main.py) via HTTP.
# Falls back to local HuggingFace ML execution if API is unreachable.
# ============================================================

import sys
import time
import json
import requests
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Page config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title  = "EmotiLearn",
    page_icon   = "🧠",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Constants ────────────────────────────────────────────────
API_BASE = "http://localhost:8000"
EMOTION_COLORS = {
    "happy":      "#22c55e",   # green
    "neutral":    "#94a3b8",   # slate
    "confused":   "#f59e0b",   # amber
    "frustrated": "#ef4444",   # red
    "bored":      "#6366f1",   # indigo
    "engaged":    "#06b6d4",   # cyan
}
EMOTION_EMOJIS = {
    "happy": "😊", "neutral": "😐", "confused": "😕",
    "frustrated": "😤", "bored": "😴", "engaged": "🤩",
}
TOPICS = {
    "general":          "📚 General",
    "recursion":        "🔄 Recursion",
    "machine_learning": "🤖 Machine Learning",
}


# ── Intelligent Local Model Cache ─────────────────────────────
@st.cache_resource
def get_local_analyzer():
    """
    Initializes the lightweight model pipelines on CPU directly inside 
    Streamlit resource cache to maintain state across page refreshes.
    """
    from transformers import pipeline
    class LocalUnifiedAnalyzer:
        def __init__(self):
            # Efficiently shares system CPU footprint
            self.voice_to_text = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")
            self.emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)
    return LocalUnifiedAnalyzer()

# ── CSS injection ─────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* ── Base ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; font-size: 0.8rem; }

    /* Main background */
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }

    /* Emotion badge */
    .emotion-badge {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 8px 18px; border-radius: 100px;
        font-family: 'DM Mono', monospace;
        font-size: 0.95rem; font-weight: 500;
        letter-spacing: 0.05em; text-transform: uppercase;
        box-shadow: 0 0 20px currentColor;
        animation: pulse-badge 2s infinite;
    }
    @keyframes pulse-badge {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }

    /* Tutor card */
    .tutor-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    .tutor-card::before {
        content: '';
        position: absolute; top: 0; left: 0;
        width: 4px; height: 100%;
        background: linear-gradient(180deg, #06b6d4, #6366f1);
    }

    /* Metric card */
    .metric-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'DM Mono', monospace;
        font-size: 2rem; font-weight: 500;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.75rem; color: #64748b;
        text-transform: uppercase; letter-spacing: 0.08em;
        margin-top: 4px;
    }

    /* Prob bar */
    .prob-row { margin: 4px 0; }
    .prob-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem; color: #94a3b8;
    }
    .prob-bar-bg {
        background: #1e293b; border-radius: 4px;
        height: 8px; overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%; border-radius: 4px;
        transition: width 0.4s ease;
    }

    /* Activity box */
    .activity-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .activity-prompt {
        font-size: 1.05rem; color: #e2e8f0;
        font-weight: 500; margin-bottom: 0.8rem;
    }
    .activity-option {
        background: #0f172a; border: 1px solid #334155;
        border-radius: 8px; padding: 0.5rem 1rem;
        margin: 0.3rem 0; cursor: pointer;
        transition: border-color 0.2s;
        font-size: 0.9rem; color: #cbd5e1;
    }
    .activity-option:hover { border-color: #06b6d4; }

    /* Timeline entry */
    .timeline-entry {
        display: flex; align-items: flex-start; gap: 12px;
        padding: 10px 0; border-bottom: 1px solid #1e293b;
    }
    .timeline-dot {
        width: 10px; height: 10px; border-radius: 50%;
        margin-top: 5px; flex-shrink: 0;
    }
    .timeline-text { font-size: 0.85rem; color: #94a3b8; }
    .timeline-emotion { font-weight: 600; }

    /* Header */
    .app-header {
        display: flex; align-items: center; gap: 14px;
        margin-bottom: 1.5rem;
    }
    .app-title {
        font-family: 'DM Mono', monospace;
        font-size: 1.6rem; font-weight: 500;
        background: linear-gradient(90deg, #06b6d4, #6366f1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    </style>
    """, unsafe_allow_html=True)



# ── Session state init ────────────────────────────────────────
def init_session_state():
    defaults = {
        "session_id":      None,
        "turn_history":    [],
        "current_emotion": "neutral",
        "current_confidence": 0.0,
        "current_probs":   {l: 1/6 for l in EMOTION_COLORS},
        "tutor_response":  None,
        "difficulty":      5,
        "topic":           "general",
        "audio_weight":    0.6,
        "api_available":   None,
        "total_turns":     0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── API helpers ───────────────────────────────────────────────
def check_api():
    """Check if FastAPI backend is reachable."""
    if st.session_state.api_available is not None:
        return st.session_state.api_available
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        st.session_state.api_available = r.status_code == 200
    except Exception:
        st.session_state.api_available = False
    return st.session_state.api_available


def api_start_session(topic: str, difficulty: int, audio_weight: float) -> Optional[str]:
    try:
        r = requests.post(f"{API_BASE}/session/start", json={
            "topic": topic,
            "initial_difficulty": difficulty,
            "audio_weight": audio_weight,
        }, timeout=5)
        if r.status_code == 200:
            return r.json()["session_id"]
    except Exception:
        pass
    return None


def api_analyze_text(text: str, session_id: str) -> Optional[Dict]:
    try:
        r = requests.post(f"{API_BASE}/analyze/text", json={
            "text": text,
            "session_id": session_id,
        }, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
    return None


def api_get_analytics(session_id: str) -> Optional[Dict]:
    try:
        r = requests.get(f"{API_BASE}/analytics/{session_id}", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def direct_analyze_text(text: str) -> Dict:
    """
    Executes actual real-time inference using the integrated local model architecture.
    Features an updated, balanced label map and a low-confidence neutral threshold fallback.
    """
    t_start = time.time()
    
    # 1. Gather local inference pipelines from resource cache
    analyzer = get_local_analyzer()
    
    # 2. Extract RAW transformer multi-class score probabilities
    raw_predictions = analyzer.emotion_classifier(text)[0]
    
    # 3. New Balanced Routing Logic
    LABEL_MAP = {
        "joy": "happy", 
        "sadness": "bored",        # Passive student state
        "anger": "frustrated",
        "fear": "confused",         # Active cognitive block
        "surprise": "happy",       # Unexpected breakthrough / pleasant surprise
        "love": "engaged"          # Deep focus / fascination
    }
    
    # Define a confidence floor. If the top model prediction is below 40%, 
    # the user's expression is ambiguous enough to be treated as baseline "neutral".
    CONFIDENCE_THRESHOLD = 0.40
    
    # Find the single highest raw scoring emotion from the transformer model
    top_raw_pred = max(raw_predictions, key=lambda x: x['score'])
    
    # Initialize all application UI metrics to a default baseline margin
    probs = {e: 0.01 for e in EMOTION_COLORS} 
    
    if top_raw_pred['score'] < CONFIDENCE_THRESHOLD:
        # Scenario A: The model is highly split or uncertain -> Mark explicitly as Neutral
        emotion = "neutral"
        probs["neutral"] = round(top_raw_pred['score'], 3)
    else:
        # Scenario B: Secure detection -> Map to your targeted app state
        emotion = LABEL_MAP.get(top_raw_pred['label'], "neutral")
        
    # Accumulate all raw scores into our UI labels for the horizontal bar charts
    for pred in raw_predictions:
        mapped_label = LABEL_MAP.get(pred['label'], "neutral")
        
        # If the prediction already fell back to a global neutral state, aggregate there
        if emotion == "neutral":
            probs["neutral"] += pred['score']
        elif mapped_label in probs:
            probs[mapped_label] += pred['score']
            
    # Normalize distribution totals cleanly to exactly 1.0 for the frontend renderers
    total_p = sum(probs.values())
    probs = {k: round(v / total_p, 3) for k, v in probs.items()}
    
    # Ensure the designated dominant state holds the exact calculated win percentage
    confidence = probs[emotion]

    # 4. Process Adaptive Tutoring Engine responses dynamically
    STRATEGIES = {
        "happy":      {"strategy":"reinforce_and_advance","message":"Great enthusiasm! Here's something more challenging:","emoji":"😊","delta":+1},
        "neutral":    {"strategy":"maintain_pace","message":"Let's continue. Here's the next concept:","emoji":"😐","delta":0},
        "confused":   {"strategy":"simplify_and_reteach","message":"No worries — let me break this down simply:","emoji":"😕","delta":-1},
        "frustrated": {"strategy":"step_by_step_breakdown","message":"Let's take it one step at a time:","emoji":"😤","delta":-1},
        "bored":      {"strategy":"gamify_and_engage","message":"Let's make this more interesting! Try this challenge:","emoji":"😴","delta":0},
        "engaged":    {"strategy":"deepen_and_challenge","message":"You're in the zone! Here's the advanced version:","emoji":"🤩","delta":+2},
    }

    CONTENT_MAP = {
        "happy": "Fantastic job mastering recursion! Since your confidence is high, let's look at how we can optimize memory space using Tail Call Optimization.",
        "neutral": "The execution stack tracks active subroutines. Let's move on to analyzing the base case requirements for tree traversal algorithms.",
        "confused": "Think of recursion like Russian nesting dolls. You can't reach the smallest, final doll (the base case) without opening all the outer ones first.",
        "frustrated": "Take a deep breath! Let's ignore the code entirely for a moment. If I ask you to count down from 5 to 1, what is the very last number you stop at?",
        "bored": "You seem to have the basics completely down. Let's fast-track this: Can you spot the hidden infinite loop vulnerability in this production-grade snippet?",
        "engaged": "Brilliant follow-up question. When we pass a complex object into a recursive branch, Python passes it by reference. Let's map out the memory address changes!",
    }

    strat = STRATEGIES[emotion]
    diff = max(1, min(10, st.session_state.difficulty + strat["delta"]))
    latency_ms = (time.time() - t_start) * 1000

    return {
        "text_emotion": {
            "emotion": emotion, "confidence": confidence,
            "probabilities": probs, "latency_ms": latency_ms, "token_count": len(text.split()),
        },
        "fusion": {
            "emotion": emotion, "confidence": confidence,
            "probabilities": probs,
            "audio_emotion": "uniform", "text_emotion": emotion,
            "models_agree": True,
            "fusion_weights": {"audio": 0.0, "text": 1.0},
            "conflict_resolved": True,
            "explanation": f"Local Inference: {emotion.upper()} ({confidence:.0%})",
        },
        "tutor": {
            "emotion": emotion,
            "strategy_name": strat["strategy"],
            "opening_message": strat["message"],
            "content": CONTENT_MAP[emotion],
            "activity": {
                "activity_type": "micro_checkpoint" if emotion not in ["frustrated", "neutral"] else "reflection",
                "prompt": f"Ready to verify this {emotion} state scenario?",
                "options": ["Let's do it!", "Show me more details"] if emotion in ["happy", "engaged", "bored"] else [], 
                "answer": None, "hint": "Look closely at the UI element response values."
            },
            "difficulty_level": diff,
            "emoji": strat["emoji"],
            "topic": st.session_state.topic,
        },
        "total_latency_ms": latency_ms,
        "session_id": st.session_state.session_id or "local",
    }
# ── UI components ─────────────────────────────────────────────
def render_emotion_badge(emotion: str, confidence: float):
    color = EMOTION_COLORS.get(emotion, "#94a3b8")
    emoji = EMOTION_EMOJIS.get(emotion, "😐")
    st.markdown(f"""
    <div style="text-align:center; margin: 1rem 0;">
        <span class="emotion-badge" style="background:{color}22; color:{color}; border: 1px solid {color}44;">
            {emoji} {emotion.upper()} &nbsp;·&nbsp; {confidence:.0%}
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_prob_bars(probs: Dict[str, float]):
    st.markdown("<div style='margin-top:0.5rem'>", unsafe_allow_html=True)
    for emotion, prob in sorted(probs.items(), key=lambda x: -x[1]):
        color = EMOTION_COLORS.get(emotion, "#64748b")
        pct = int(prob * 100)
        bar_width = int(prob * 100)
        st.markdown(f"""
        <div class="prob-row">
          <div style="display:flex; justify-content:space-between; margin-bottom:2px">
            <span class="prob-label">{EMOTION_EMOJIS.get(emotion,'')} {emotion}</span>
            <span class="prob-label">{pct}%</span>
          </div>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{bar_width}%; background:{color}"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_tutor_card(tutor: Dict):
    emoji = tutor.get("emoji", "🤖")
    opening = tutor.get("opening_message", "")
    content = tutor.get("content", "")
    strategy = tutor.get("strategy_name", "")
    difficulty = tutor.get("difficulty_level", 5)

    st.markdown(f"""
    <div class="tutor-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem">
            <span style="font-size:0.7rem; color:#475569; text-transform:uppercase; letter-spacing:0.1em; font-family:'DM Mono',monospace">
                🎯 {strategy.replace('_',' ')}
            </span>
            <span style="font-family:'DM Mono',monospace; font-size:0.75rem; color:#475569">
                LEVEL {difficulty}/10
            </span>
        </div>
        <div style="font-size:1.05rem; color:#e2e8f0; font-weight:500; margin-bottom:0.6rem">
            {emoji} {opening}
        </div>
        <div style="font-size:0.9rem; color:#94a3b8; line-height:1.7; white-space:pre-line">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Activity
    activity = tutor.get("activity")
    if activity and activity.get("prompt"):
        render_activity(activity)


def render_activity(activity: Dict):
    atype = activity.get("activity_type", "open_question")
    prompt = activity.get("prompt", "")
    options = activity.get("options", [])
    hint = activity.get("hint")

    icon_map = {
        "multiple_choice": "🔘", "open_question": "💬",
        "step_by_step": "📋", "analogy": "🔗",
        "challenge": "⚡", "checkpoint": "✅",
        "micro_checkpoint": "🔍", "reflection": "💭",
    }
    icon = icon_map.get(atype, "📝")

    st.markdown(f"""
    <div class="activity-box">
        <div style="font-size:0.7rem; color:#475569; text-transform:uppercase;
                    letter-spacing:0.1em; margin-bottom:0.6rem; font-family:'DM Mono',monospace">
            {icon} {atype.replace('_',' ')}
        </div>
        <div class="activity-prompt">{prompt}</div>
    """, unsafe_allow_html=True)

    if options:
        for opt in options:
            st.markdown(f'<div class="activity-option">→ {opt}</div>',
                        unsafe_allow_html=True)

    if hint:
        st.markdown(f"""
        <div style="margin-top:0.8rem; padding:0.6rem 0.8rem; background:#0f172a;
                    border-radius:8px; border-left:3px solid #f59e0b">
            <span style="font-size:0.8rem; color:#f59e0b">💡 Hint:</span>
            <span style="font-size:0.82rem; color:#94a3b8; margin-left:6px">{hint}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_difficulty_bar(level: int):
    filled = "█" * level
    empty  = "░" * (10 - level)
    color  = "#ef4444" if level <= 3 else "#f59e0b" if level <= 6 else "#22c55e"
    st.markdown(f"""
    <div style="margin: 0.3rem 0">
        <span style="font-family:'DM Mono',monospace; font-size:0.85rem; color:{color}">
            {filled}<span style='color:#1e293b'>{empty}</span>
        </span>
        <span style="font-family:'DM Mono',monospace; font-size:0.75rem;
                     color:#475569; margin-left:8px">{level}/10</span>
    </div>
    """, unsafe_allow_html=True)


def render_timeline(history: List[Dict]):
    if not history:
        st.markdown("<p style='color:#475569; font-size:0.85rem'>No turns yet.</p>",
                    unsafe_allow_html=True)
        return
    for entry in reversed(history[-10:]):
        emotion = entry.get("emotion", "neutral")
        color   = EMOTION_COLORS.get(emotion, "#94a3b8")
        emoji   = EMOTION_EMOJIS.get(emotion, "")
        conf    = entry.get("confidence", 0)
        ts      = entry.get("timestamp", 0)
        transcript = entry.get("transcript", "")[:60]
        agree   = "✓" if entry.get("models_agree") else "~"
        st.markdown(f"""
        <div class="timeline-entry">
            <div class="timeline-dot" style="background:{color}"></div>
            <div>
                <span class="timeline-emotion" style="color:{color}">{emoji} {emotion}</span>
                <span style="color:#475569; font-size:0.78rem; margin-left:8px">
                    {conf:.0%} · {ts:.1f}s · {agree}
                </span>
                <div class="timeline-text">"{transcript}{'...' if len(transcript)==60 else ''}"</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem 0 1.5rem 0">
            <div style="font-family:'DM Mono',monospace; font-size:1.2rem;
                         background:linear-gradient(90deg,#06b6d4,#6366f1);
                         -webkit-background-clip:text; -webkit-text-fill-color:transparent">
                🧠 EmotiLearn
            </div>
            <div style="font-size:0.72rem; color:#475569; margin-top:2px">
                Emotion-Aware Adaptive Tutor
            </div>
        </div>
        """, unsafe_allow_html=True)

        # API status
        api_ok = check_api()
        status_color = "#22c55e" if api_ok else "#f59e0b"
        status_text  = "API Online" if api_ok else "Direct Mode (CPU)"
        status_icon  = "●" if api_ok else "○"
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace; font-size:0.72rem;
                    color:{status_color}; margin-bottom:1.5rem">
            {status_icon} {status_text}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigate",
            ["🎓 Student View", "📊 Analytics", "ℹ️ About"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("<div style='font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>Session Config</div>", unsafe_allow_html=True)

        topic = st.selectbox(
            "Topic",
            options=list(TOPICS.keys()),
            format_func=lambda x: TOPICS[x],
        )
        difficulty = st.slider("Initial Difficulty", 1, 10, 5)
        audio_weight = st.slider("Audio Weight (α)", 0.1, 0.9, 0.6, 0.1,
                                 help="Higher = trust audio model more")

        if st.button("▶ New Session", use_container_width=True, type="primary"):
            st.session_state.turn_history    = []
            st.session_state.current_emotion = "neutral"
            st.session_state.current_confidence = 0.0
            st.session_state.current_probs   = {l: 1/6 for l in EMOTION_COLORS}
            st.session_state.tutor_response  = None
            st.session_state.difficulty      = difficulty
            st.session_state.topic           = topic
            st.session_state.audio_weight    = audio_weight
            st.session_state.total_turns     = 0

            if api_ok:
                sid = api_start_session(topic, difficulty, audio_weight)
                st.session_state.session_id = sid
            else:
                st.session_state.session_id = f"local-{int(time.time())}"
            st.success("Session started!")
            st.rerun()

        # Current difficulty indicator
        if st.session_state.turn_history:
            st.markdown("---")
            st.markdown("<div style='font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem'>Current Difficulty</div>", unsafe_allow_html=True)
            render_difficulty_bar(st.session_state.difficulty)

    return page


# ── Page: Student View ────────────────────────────────────────
def page_student():
    st.markdown("""
    <div class="app-header">
        <div>
            <div class="app-title">EmotiLearn</div>
            <div style="font-size:0.8rem; color:#475569; margin-top:2px">
                Adaptive Tutoring · Real-Time Emotion Detection
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.session_id:
        st.info("👈  Click **▶ New Session** in the sidebar to begin.")
        return

    col_left, col_right = st.columns([1, 1.4], gap="large")

    with col_left:
        st.markdown("<div style='font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>Your Response</div>", unsafe_allow_html=True)

        tab_text, tab_audio = st.tabs(["✏️ Text", "🎙️ Audio"])

        with tab_text:
            user_text = st.text_area(
                "Type your response:",
                placeholder="e.g. 'I don't understand why the base case is needed...'",
                height=120,
                label_visibility="collapsed",
            )
            col_btn, col_clr = st.columns([3, 1])
            with col_btn:
                submit = st.button("Analyze →", use_container_width=True, type="primary")
            with col_clr:
                if st.button("Clear", use_container_width=True):
                    st.rerun()

        with tab_audio:
            st.markdown("""
            <div style="padding:1.5rem; text-align:center; background:#0f172a;
                        border:1px dashed #334155; border-radius:12px; color:#475569">
                <div style="font-size:2rem; margin-bottom:0.5rem">🎙️</div>
                <div style="font-size:0.85rem">Audio local pipeline ready</div>
                <div style="font-size:0.75rem; margin-top:4px; color:#334155">
                    Upload a WAV/MP3 file for full Whisper transcription
                </div>
            </div>
            """, unsafe_allow_html=True)

            audio_file = st.file_uploader(
                "Upload audio (WAV/MP3/FLAC)",
                type=["wav","mp3","flac","ogg"],
                label_visibility="collapsed",
            )
            if audio_file:
                st.audio(audio_file)
                submit_audio = st.button("Analyze Audio →", type="primary",
                                          use_container_width=True)
            else:
                submit_audio = False

        # ── Process submission ────────────────────────────────
        result = None

        if submit and user_text.strip():
            with st.spinner("Running local Transformer pipeline..."):
                if check_api() and st.session_state.session_id:
                    result = api_analyze_text(user_text, st.session_state.session_id)
                else:
                    result = direct_analyze_text(user_text)

            if result:
                _update_state(result, user_text)
                st.rerun()

        elif submit_audio and audio_file:
            with st.spinner("Processing audio engine via local Whisper..."):
                if check_api():
                    try:
                        files = {"audio_file": (audio_file.name, audio_file, audio_file.type)}
                        data  = {"session_id": st.session_state.session_id, "topic": st.session_state.topic}
                        r = requests.post(f"{API_BASE}/analyze", files=files, data=data, timeout=20)
                        if r.status_code == 200:
                            result = r.json()
                            _update_state(result, result.get("transcription",{}).get("text",""))
                            st.rerun()
                        else:
                            st.error(f"API error {r.status_code}: {r.text[:200]}")
                    except Exception as e:
                        st.error(f"Audio analysis failed: {e}")
                else:
                    # Direct mode local execution of audio files via Whisper tiny
                    try:
                        import tempfile
                        import os
                        analyzer = get_local_analyzer()
                        
                        # Write streamlit uploaded file binary buffer safely to text chunk path
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                            tmpfile.write(audio_file.read())
                            tmpfile_path = tmpfile.name
                        
                        try:
                            asr_result = analyzer.voice_to_text(tmpfile_path)
                            transcribed_text = asr_result["text"].strip()
                            
                            if transcribed_text:
                                result = direct_analyze_text(transcribed_text)
                                _update_state(result, transcribed_text)
                                st.rerun()
                            else:
                                st.error("Whisper execution returned empty transcription matrix.")
                        finally:
                            if os.path.exists(tmpfile_path):
                                os.remove(tmpfile_path)
                    except Exception as e:
                        st.error(f"Local ASR Processing Fault: {e}")

        # ── Emotion display ───────────────────────────────────
        st.markdown("---")
        st.markdown("<div style='font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>Detected Emotion</div>", unsafe_allow_html=True)

        render_emotion_badge(
            st.session_state.current_emotion,
            st.session_state.current_confidence,
        )
        render_prob_bars(st.session_state.current_probs)

        if st.session_state.turn_history:
            st.markdown("---")
            st.markdown("<div style='font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem'>Turn History</div>", unsafe_allow_html=True)
            render_timeline(st.session_state.turn_history)

    with col_right:
        st.markdown("<div style='font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>Adaptive Tutor Response</div>", unsafe_allow_html=True)

        if st.session_state.tutor_response:
            render_tutor_card(st.session_state.tutor_response)

            if "total_latency_ms" in (st.session_state.get("last_result") or {}):
                lat = st.session_state.last_result["total_latency_ms"]
                st.markdown(f"""
                <div style="text-align:right; font-family:'DM Mono',monospace;
                            font-size:0.7rem; color:#334155; margin-top:0.5rem">
                    ⚡ {lat:.0f}ms
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="padding:3rem 2rem; text-align:center;
                        background:#0f172a; border:1px dashed #1e293b;
                        border-radius:16px; color:#334155">
                <div style="font-size:3rem; margin-bottom:1rem">🤖</div>
                <div style="font-size:0.9rem; color:#475569">
                    Your tutor is ready.<br>
                    <span style="font-size:0.8rem">Type something to begin →</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.turn_history:
            st.markdown("---")
            h = st.session_state.turn_history
            from collections import Counter
            counts = Counter(e["emotion"] for e in h)
            dom = counts.most_common(1)[0][0]
            avg_conf = sum(e["confidence"] for e in h) / len(h)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:{EMOTION_COLORS.get(dom,'#94a3b8')}">{EMOTION_EMOJIS.get(dom,'')}</div>
                    <div class="metric-label">Dominant</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:#06b6d4">{len(h)}</div>
                    <div class="metric-label">Turns</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:#6366f1">{avg_conf:.0%}</div>
                    <div class="metric-label">Avg Conf</div>
                </div>""", unsafe_allow_html=True)


def _update_state(result: Dict, text: str):
    fusion = result.get("fusion", result.get("text_emotion", {}))
    tutor  = result.get("tutor", {})
    emotion    = fusion.get("emotion", "neutral")
    confidence = fusion.get("confidence", 0.5)
    probs = fusion.get("probabilities", {l: 1/6 for l in EMOTION_COLORS})

    st.session_state.current_emotion    = emotion
    st.session_state.current_confidence = confidence
    st.session_state.current_probs      = probs
    st.session_state.tutor_response     = tutor
    st.session_state.difficulty         = tutor.get("difficulty_level", st.session_state.difficulty)
    st.session_state.total_turns       += 1
    st.session_state.last_result        = result

    st.session_state.turn_history.append({
        "timestamp":    time.time(),
        "emotion":      emotion,
        "confidence":   confidence,
        "audio_emotion": fusion.get("audio_emotion","—"),
        "text_emotion":  fusion.get("text_emotion", emotion),
        "models_agree":  fusion.get("models_agree", False),
        "transcript":    text[:100],
    })


# ── Page: Analytics Dashboard ─────────────────────────────────
def page_analytics():
    st.markdown("<div class='app-title' style='margin-bottom:1.5rem'>📊 Analytics Dashboard</div>",
                unsafe_allow_html=True)

    if not st.session_state.turn_history:
        st.info("No session data yet. Complete some turns in Student View first.")
        return

    history = st.session_state.turn_history
    from collections import Counter

    emotions = [h["emotion"] for h in history]
    counts   = Counter(emotions)
    dom      = counts.most_common(1)[0][0]
    avg_conf = sum(h["confidence"] for h in history) / len(history)
    agree    = sum(1 for h in history if h.get("models_agree")) / len(history)

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (col1, EMOTION_EMOJIS.get(dom,""), "Dominant", EMOTION_COLORS.get(dom,"#06b6d4")),
        (col2, len(history), "Turns",      "#06b6d4"),
        (col3, f"{avg_conf:.0%}", "Avg Confidence", "#6366f1"),
        (col4, f"{agree:.0%}", "Agreement Rate", "#22c55e"),
    ]
    for col, val, label, color in metrics:
        with col:
            st.markdown(f"""<div class="metric-card" style="border-color:{color}22">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_left:
        st.markdown("<div style='font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem'>Emotion Distribution</div>", unsafe_allow_html=True)

        total = len(history)
        sorted_emotions = sorted(counts.items(), key=lambda x: -x[1])
        for emotion, count in sorted_emotions:
            color = EMOTION_COLORS.get(emotion, "#64748b")
            pct   = count / total
            bar   = int(pct * 100)
            st.markdown(f"""
            <div style="margin:6px 0">
                <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                    <span style="font-size:0.85rem;color:#e2e8f0">
                        {EMOTION_EMOJIS.get(emotion,'')} {emotion}
                    </span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.8rem;color:#475569">
                        {count} ({pct:.0%})
                    </span>
                </div>
                <div style="background:#1e293b;border-radius:6px;height:12px;overflow:hidden">
                    <div style="width:{bar}%;background:{color};height:100%;
                                border-radius:6px;transition:width 0.4s"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if len(history) > 1:
            st.markdown("<div style='font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-top:1.5rem;margin-bottom:0.8rem'>Difficulty Trend</div>", unsafe_allow_html=True)
            try:
                difficulties = [h.get("difficulty", st.session_state.difficulty) for h in history]
                import pandas as pd
                df = pd.DataFrame({
                    "Turn": list(range(1, len(difficulties)+1)),
                    "Difficulty": difficulties,
                })
                st.line_chart(df.set_index("Turn"), height=180, use_container_width=True)
            except Exception:
                diffs = [str(h.get("difficulty", "?")) for h in history]
                st.markdown(f"""<div style="font-family:'DM Mono',monospace;
                    font-size:0.8rem;color:#94a3b8">
                    {' → '.join(diffs)}</div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div style='font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>Full Timeline</div>", unsafe_allow_html=True)
        render_timeline(history)

        st.markdown("<div style='font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-top:1.2rem;margin-bottom:0.5rem'>Model Agreement</div>", unsafe_allow_html=True)
        agreed    = sum(1 for h in history if h.get("models_agree", False))
        disagreed = total - agreed
        st.markdown(f"""
        <div style="display:flex;gap:12px;margin-top:0.3rem">
            <div class="metric-card" style="flex:1">
                <div class="metric-value" style="color:#22c55e;font-size:1.4rem">{agreed}</div>
                <div class="metric-label">Agreed</div>
            </div>
            <div class="metric-card" style="flex:1">
                <div class="metric-value" style="color:#f59e0b;font-size:1.4rem">{disagreed}</div>
                <div class="metric-label">Diverged</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("⬇️ Export Session JSON"):
        export = {
            "session_id":  st.session_state.session_id,
            "topic":       st.session_state.topic,
            "turns":       len(history),
            "history":     history,
            "summary": {
                "dominant_emotion": dom,
                "emotion_counts":   dict(counts),
                "avg_confidence":   round(avg_conf, 3),
                "agreement_rate":   round(agree, 3),
            }
        }
        st.download_button(
            "Download JSON",
            data=json.dumps(export, indent=2, default=str),
            file_name=f"emotilearn_session_{int(time.time())}.json",
            mime="application/json",
        )


# ── Page: About ───────────────────────────────────────────────
def page_about():
    st.markdown("<div class='app-title' style='margin-bottom:1.5rem'>ℹ️ About EmotiLearn</div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div class="tutor-card">
            <div style="font-size:0.7rem;color:#475569;text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:0.8rem">System Architecture</div>
            <div style="font-size:0.9rem;color:#94a3b8;line-height:1.8">
                🎤 <strong style="color:#e2e8f0">Whisper Tiny</strong> — Speech-to-text (39MB, ~500ms)<br>
                🔊 <strong style="color:#e2e8f0">Audio CNN</strong> — MFCC features + 3-layer CNN (~30ms)<br>
                📝 <strong style="color:#e2e8f0">DistilBERT</strong> — Text emotion, INT8 quantized (~250ms)<br>
                🔀 <strong style="color:#e2e8f0">Late Fusion</strong> — Weighted α=0.6/β=0.4 + dynamic adj.<br>
                🧠 <strong style="color:#e2e8f0">Interpreter</strong> — Streak detection + escalation rules<br>
                📚 <strong style="color:#e2e8f0">Adaptive Tutor</strong> — Template-based, &lt;5ms response
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="tutor-card">
            <div style="font-size:0.7rem;color:#475569;text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:0.8rem">Emotion Classes</div>
        """, unsafe_allow_html=True)
        for emotion, color in EMOTION_COLORS.items():
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin:6px 0">
                <div style="width:10px;height:10px;border-radius:50%;
                            background:{color};flex-shrink:0"></div>
                <span style="font-size:0.85rem;color:#e2e8f0;text-transform:capitalize">{emotion}</span>
                <span style="font-size:0.75rem;color:#475569;margin-left:auto">
                    {EMOTION_EMOJIS.get(emotion,'')}
                </span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.5rem;padding:1rem 1.5rem;background:#0f172a;
                border-radius:12px;border:1px solid #1e293b">
        <div style="font-size:0.7rem;color:#475569;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:0.5rem">Run Commands</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.8rem;color:#94a3b8;line-height:2">
            <span style="color:#06b6d4">$</span> uvicorn api.main:app --port 8000<br>
            <span style="color:#06b6d4">$</span> streamlit run app/streamlit_app.py<br>
            <span style="color:#06b6d4">$</span> python training/train_audio_cnn.py --synthetic<br>
            <span style="color:#06b6d4">$</span> python training/finetune_distilbert.py --quick-test<br>
            <span style="color:#06b6d4">$</span> pytest tests/ -v
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────
def main():
    inject_css()
    init_session_state()
    page = render_sidebar()

    if page == "🎓 Student View":
        page_student()
    elif page == "📊 Analytics":
        page_analytics()
    else:
        page_about()


if __name__ == "__main__":
    main()