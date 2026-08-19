import streamlit as st
import plotly.graph_objects as go
import json
import re
from page_config import render_page_config

render_page_config()

# Initialize session state keys so this page also works when opened directly
# (without visiting the home page first).
if "response" not in st.session_state:
    st.session_state.response = None
if "feedback_response" not in st.session_state:
    st.session_state.feedback_response = None

def _safe_score(value):
    """Return value if it is a number between 0 and 5, otherwise 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0.0, min(float(value), 5.0))


def _string_list(value):
    """Return value if it is a list of strings, otherwise an empty list."""
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return []


def _extract_json_text(text):
    """Best-effort extraction of a JSON object from model text."""
    candidates = [text.strip()]
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())
    start = text.find("{")
    if start != -1:
        end = text.rfind("}")
        if end > start:
            candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


SCORE_LABELS = {
    "content_organization": "Content & Organization",
    "delivery_vocal_quality": "Delivery & Vocal Quality",
    "body_language_eye_contact": "Body Language & Eye Contact",
    "audience_engagement": "Audience Engagement",
    "language_clarity": "Language & Clarity",
}

# Get feedback response from session state
feedback_response = None
if st.session_state.feedback_response:
    feedback_response = _extract_json_text(st.session_state.feedback_response)
    if feedback_response is None:
        st.warning("Feedback data could not be parsed. Please run the analysis again.")

if isinstance(feedback_response, dict):
    feedback_scores = feedback_response.get("scores")
    if not isinstance(feedback_scores, dict):
        feedback_scores = {}
    scores = {label: _safe_score(feedback_scores.get(key)) for key, label in SCORE_LABELS.items()}

    try:
        total_score = int(feedback_response.get("total_score"))
    except (TypeError, ValueError):
        total_score = sum(scores.values())

    interpretation = feedback_response.get("interpretation")
    interpretation = interpretation if isinstance(interpretation, str) else ""
    feedback_summary = feedback_response.get("feedback_summary")
    feedback_summary = feedback_summary if isinstance(feedback_summary, str) else ""
else:
    st.warning("No feedback available! Please upload a video and analyze it first.")
    scores = {label: 0 for label in SCORE_LABELS.values()}
    total_score = 0
    interpretation = ""
    feedback_summary = ""

# Calculate average score
average_score = sum(scores.values()) / len(scores)

# Determine strengths, weaknesses, and suggestions for improvement
response = st.session_state.response if isinstance(st.session_state.response, dict) else {}
strengths = _string_list(response.get("strengths"))
weaknesses = _string_list(response.get("weaknesses"))
suggestions = _string_list(response.get("suggestions"))

# Create three columns with equal width
col1, col2, col3 = st.columns([0.3, 0.4, 0.3])

# Left Column: Evaluation Summary
with col1:
    st.subheader("🧾 Evaluation Summary")

    st.markdown("<br>", unsafe_allow_html=True)

    for criterion, score in scores.items():
        label_col, progress_col, score_col = st.columns([2, 3, 1])  # Adjust the ratio as needed
        with label_col:
            st.markdown(f"**{criterion}**")
        with progress_col:
            st.progress(score / 5)
        with score_col:
            st.markdown(f"<span><b>{score}/5</b></span>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Display total score
    st.markdown(f"#### 🏆 Total Score: {total_score} / 25")
    # Display average score
    st.markdown(f"#### 🎯 Average Score: {average_score:.2f} / 5")

    st.markdown("""---""")

    st.markdown("##### 🗣️ Feedback Summary:")
    # Display interpretation
    st.markdown(f"📝 **Overall Assessment**: {interpretation}")
    # Display feedback summary
    st.info(f"{feedback_summary}")

# Middle Column: Strengths, Weaknesses, and Suggestions
with col2:
    # Display strengths
    st.markdown("##### 🦾 Strengths:")
    strengths_text = '\n'.join(f"- {item}" for item in strengths)
    st.success(strengths_text)

    # Display weaknesses
    st.markdown("##### ⚠️ Weaknesses:")
    weaknesses_text = '\n'.join(f"- {item}" for item in weaknesses)
    st.error(weaknesses_text)

    # Display suggestions
    st.markdown("##### 💡 Suggestions for Improvement:")
    suggestions_text = '\n'.join(f"- {item}" for item in suggestions)
    st.warning(suggestions_text)

# Right Column: Performance Chart
with col3:
    st.subheader("📊 Performance Chart")

    # Radar Chart
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=list(scores.values()),
        theta=list(scores.keys()),
        fill='toself',
        name='Scores'
    ))
    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5])
        ),
        showlegend=False,
        margin=dict(t=50, b=50, l=50, r=50),  # Reduced margins
        width=350,
        height=350
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    st.markdown("""---""")
