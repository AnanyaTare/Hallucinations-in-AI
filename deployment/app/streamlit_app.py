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
    st.subheader("📜 Conversation History")
    st.caption("All previous interactions stored in the database — questions, answers, and hallucination scores.")

    # ── Controls row ─────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1, 1, 1])
    with ctrl1:
        search_term = st.text_input("🔍 Search conversations", placeholder="Search by keyword...")
    with ctrl2:
        filter_model = st.selectbox("Filter by Model", ["All", "groq/llama-3.1-8b-instant",
                                    "groq/llama-3.3-70b-versatile", "gemini/gemini-2.0-flash", "gpt-4o"])
    with ctrl3:
        hallucinated_only = st.toggle("🔴 Hallucinated only", value=False)
    with ctrl4:
        limit = st.selectbox("Show", [10, 20, 50], index=0)

    col_refresh, col_export = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    st.divider()

    # ── Fetch history from API ────────────────────────────────────────────────
    try:
        params = f"?limit={limit}"
        if hallucinated_only:
            params += "&hallucinated_only=true"
        if filter_model != "All":
            params += f"&model={filter_model}"

        resp = httpx.get(f"{API_BASE}/api/history{params}", timeout=10.0)

        if resp.status_code == 200:
            data = resp.json()
            sessions = data.get("sessions", [])
            total = data.get("total_sessions", 0)

            # Client-side keyword search
            if search_term:
                sessions = [
                    s for s in sessions
                    if any(
                        search_term.lower() in msg.get("content", "").lower()
                        for msg in s.get("messages", [])
                    )
                ]

            # ── Summary stats ─────────────────────────────────────────────────
            total_messages = sum(len(s.get("messages", [])) for s in sessions)
            hallucinated_sessions = sum(
                1 for s in sessions
                if any(m.get("is_hallucinated") for m in s.get("messages", []) if m["role"] == "assistant")
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📂 Total Sessions", total)
            m2.metric("💬 Messages Shown", total_messages)
            m3.metric("🔴 Hallucinated Sessions", hallucinated_sessions)
            m4.metric("🟢 Faithful Sessions", len(sessions) - hallucinated_sessions)

            st.divider()

            # ── Session list ──────────────────────────────────────────────────
            if not sessions:
                st.info("💬 No conversations found. Start chatting in the **Chat & Detect** tab!")
            else:
                for session in sessions:
                    messages = session.get("messages", [])
                    has_hallucination = any(
                        m.get("is_hallucinated") for m in messages if m["role"] == "assistant"
                    )
                    session_badge = "🔴" if has_hallucination else "🟢"
                    session_title = session.get("session_title", "Untitled")[:70]
                    msg_count = len(messages)

                    with st.expander(
                        f"{session_badge} Session #{session['id']} — {session_title}  ({msg_count} messages)",
                        expanded=False,
                    ):
                        # Session metadata
                        s1, s2, s3 = st.columns(3)
                        s1.caption(f"🤖 **Model:** {session.get('model_name', 'N/A')}")
                        s2.caption(f"🛡️ **Technique:** {session.get('mitigation', 'none').replace('_', ' ').title()}")
                        s3.caption(f"🕒 **Updated:** {session.get('updated_at', '')[:16].replace('T', ' ')}")

                        st.markdown("---")

                        # Messages (chat format)
                        for msg in messages:
                            role = msg.get("role", "user")
                            content = msg.get("content", "")

                            if role == "user":
                                st.markdown(f"""
<div style="background:#1e2a3a; border-left:3px solid #4a90d9;
            padding:10px 14px; border-radius:8px; margin:6px 0;">
    <strong>👤 You:</strong><br>{content}
</div>""", unsafe_allow_html=True)
                            else:
                                h_score = msg.get("hallucination_score")
                                h_type  = msg.get("hallucination_type", "")
                                is_h    = msg.get("is_hallucinated", False)
                                border  = "#ff4b4b" if is_h else "#00c853"
                                badge   = "🔴 Hallucinated" if is_h else "🟢 Faithful"
                                score_txt = f" | Score: {h_score:.3f}" if h_score is not None else ""
                                type_txt  = f" | Type: {h_type.replace('_',' ').title()}" if h_type else ""

                                st.markdown(f"""
<div style="background:#1a2b1a; border-left:3px solid {border};
            padding:10px 14px; border-radius:8px; margin:6px 0;">
    <strong>🤖 AI {badge}{score_txt}{type_txt}:</strong><br>{content}
</div>""", unsafe_allow_html=True)

                        # Delete button
                        st.markdown("")
                        if st.button(f"🗑️ Delete Session #{session['id']}", key=f"del_{session['id']}"):
                            try:
                                del_resp = httpx.delete(
                                    f"{API_BASE}/api/history/{session['id']}", timeout=10.0
                                )
                                if del_resp.status_code == 204:
                                    st.success("Session deleted!")
                                    st.rerun()
                                else:
                                    st.error("Failed to delete session.")
                            except Exception as ex:
                                st.error(f"Error: {ex}")

        else:
            st.error(f"Failed to load history (status {resp.status_code}).")

    except Exception as e:
        st.error(f"Could not load history: {e}")
        st.info("Make sure the backend is running: `venv\\Scripts\\uvicorn.exe deployment.api.main:app --reload --port 8000`")

