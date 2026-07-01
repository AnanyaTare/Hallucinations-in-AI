"""
streamlit_app.py — Interactive Hallucination Detection Dashboard.
Run with: streamlit run deployment/app/streamlit_app.py
"""

import streamlit as st
import httpx
import os

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
st.set_page_config(
    page_title="Hallucination Detector",
    page_icon="🧠",
    layout="wide",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stTextArea textarea { font-size: 15px; }
    .score-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2d3247;
    }
    .hallucinated { border-color: #ff4b4b !important; }
    .faithful    { border-color: #00c853 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def call_api(endpoint: str, payload: dict) -> dict | None:
    try:
        response = httpx.post(f"{API_BASE}{endpoint}", json=payload, timeout=300.0)  # 5 min for slow first-run ML model loading
        response.raise_for_status()
        return response.json()
    except httpx.ConnectError:
        st.error("❌ Cannot connect to API server. Make sure it's running on port 8000.")
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"❌ API error {e.response.status_code}: {e.response.text}")
        return None


def get_health() -> dict | None:
    try:
        r = httpx.get(f"{API_BASE}/api/health", timeout=5.0)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    model = st.selectbox("LLM Model", [
        "groq/llama-3.1-8b-instant", "groq/llama-3.3-70b-versatile", "groq/gemma2-9b-it",
        "gemini/gemini-2.0-flash", "gpt-4o", "gpt-3.5-turbo"
    ])
    mitigation = st.selectbox("Mitigation Technique", [
        "none", "few_shot", "chain_of_thought", "rag", "self_consistency"
    ])
    evaluate_hallucination = st.toggle("Evaluate for Hallucinations", value=False)

    st.divider()
    st.subheader("🏥 API Health")
    health = get_health()
    if health:
        status = health.get("status", "unknown")
        color = "🟢" if status == "healthy" else "🟡" if status == "degraded" else "🔴"
        st.markdown(f"{color} **{status.upper()}**")
        st.caption(f"Sessions: {health.get('total_chat_sessions', 0)} | "
                   f"Evals: {health.get('total_evaluations', 0)}")
    else:
        st.markdown("🔴 **API Offline**")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🧠 Hallucination Detection in LLMs")
st.caption("U2UInnovate Project 22 — AI Safety Lab")

tab1, tab2, tab3 = st.tabs(["💬 Chat & Detect", "🔍 Analyze Text", "📊 View History"])


# ── Tab 1: Chat ──────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Send a prompt to the AI and detect hallucinations")

    prompt = st.text_area(
        "Your prompt / question",
        placeholder="e.g. When was the Eiffel Tower built?",
        height=120,
    )
    ground_truth = st.text_input(
        "Ground truth (optional — improves scoring accuracy)",
        placeholder="e.g. The Eiffel Tower was built between 1887 and 1889.",
    )

    col_send, col_clear = st.columns([1, 5])
    with col_send:
        send = st.button("▶ Send", type="primary", use_container_width=True)

    if send and prompt:
        with st.spinner("Calling LLM and evaluating..."):
            result = call_api("/api/chat", {
                "prompt": prompt,
                "ground_truth": ground_truth or None,
                "model": model,
                "mitigation": mitigation,
                "evaluate_hallucination": evaluate_hallucination,
            })

        if result:
            st.divider()
            st.subheader("🤖 AI Response")
            st.markdown(f"> {result['ai_response']}")

            st.subheader("📊 Hallucination Scores")
            c1, c2, c3, c4 = st.columns(4)

            is_h = result["is_hallucinated"]
            card_class = "hallucinated" if is_h else "faithful"
            badge = "🔴 Hallucinated" if is_h else "🟢 Faithful"

            with c1:
                st.markdown(
                    f'<div class="score-card {card_class}"><h2>{badge}</h2></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                score = result["hallucination_score"]
                st.metric("Hallucination Score", f"{score:.3f}", help="0=faithful, 1=hallucinated")
            with c3:
                st.metric("Type", result["hallucination_type"].replace("_", " ").title())
            with c4:
                st.metric("Technique", mitigation.replace("_", " ").title())

            if result.get("explanation"):
                st.info(f"💬 **Judge's Reasoning:** {result['explanation']}")

            with st.expander("🔎 Full Response Details"):
                st.json(result)


# ── Tab 2: Analyze ───────────────────────────────────────────────────────────
with tab2:
    st.subheader("Analyze any piece of text for hallucination patterns")

    text_input = st.text_area("Text to analyze", height=150,
                               placeholder="Paste any AI-generated text here...")
    reference_input = st.text_input("Reference / ground truth (optional)",
                                     placeholder="Known correct information...")

    if st.button("🔍 Analyze", type="primary"):
        if text_input:
            with st.spinner("Analyzing..."):
                result = call_api("/api/v1/analyze/", {
                    "text": text_input,
                    "reference": reference_input or None,
                })
            if result:
                c1, c2, c3 = st.columns(3)
                c1.metric("Type", result["hallucination_type"].replace("_", " ").title())
                c2.metric("Confidence", f"{result['confidence']:.1%}")
                c3.metric("Semantic Similarity", f"{result['semantic_similarity']:.3f}")
                if result.get("explanation"):
                    st.info(result["explanation"])
                if result.get("contradictions_detected"):
                    st.warning("⚠️ Detected contradictions:")
                    for c in result["contradictions_detected"]:
                        st.markdown(f"- {c}")
        else:
            st.warning("Please enter some text to analyze.")


# ── Tab 3: History ───────────────────────────────────────────────────────────
with tab3:
    st.subheader("Recent conversation history")

    if st.button("🔄 Refresh"):
        st.rerun()

    try:
        resp = httpx.get(f"{API_BASE}/api/history?limit=10", timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            sessions = data.get("sessions", [])
            if not sessions:
                st.info("No conversations yet. Start chatting in the Chat tab!")
            for session in sessions:
                with st.expander(
                    f"💬 Session #{session['id']} — {session.get('session_title', 'Untitled')[:60]}",
                    expanded=False,
                ):
                    st.caption(f"Model: {session['model_name']} | Technique: {session['mitigation']}")
                    for msg in session.get("messages", []):
                        role_icon = "👤" if msg["role"] == "user" else "🤖"
                        hallu_badge = ""
                        if msg["role"] == "assistant" and msg.get("is_hallucinated"):
                            hallu_badge = " 🔴"
                        elif msg["role"] == "assistant":
                            hallu_badge = " 🟢"
                        st.markdown(f"**{role_icon} {msg['role'].title()}{hallu_badge}:** {msg['content']}")
        else:
            st.error("Failed to load history.")
    except Exception as e:
        st.error(f"Could not load history: {e}")
