import streamlit as st
import plotly.graph_objects as go
import json
from page_config import render_page_config

render_page_config()

# =========================
# ✅ 安全初始化 Session State
# =========================
if "feedback_response" not in st.session_state:
    st.session_state.feedback_response = None

if "response" not in st.session_state:
    st.session_state.response = None


def _load_json_maybe(obj):
    """obj 可以是 dict / JSON 字符串 / None，统一转 dict"""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            return {}
    return {}


# =========================
# ✅ 统一获取 feedback_response（两条路都兼容）
# =========================
raw_feedback = st.session_state.get("feedback_response")

# 如果 session_state.feedback_response 没写上，就从 response 里兜底拿
if not raw_feedback and isinstance(st.session_state.get("response"), dict):
    raw_feedback = st.session_state.response.get("feedback_response")

feedback_response = _load_json_maybe(raw_feedback)

# =========================
# 读取评分数据
# =========================
if feedback_response:
    feedback_scores = feedback_response.get("scores", {}) or {}

    scores = {
        "Content & Organization": feedback_scores.get("content_organization", 0),
        "Delivery & Vocal Quality": feedback_scores.get("delivery_vocal_quality", 0),
        "Body Language & Eye Contact": feedback_scores.get("body_language_eye_contact", 0),
        "Audience Engagement": feedback_scores.get("audience_engagement", 0),
        "Language & Clarity": feedback_scores.get("language_clarity", 0)
    }

    total_score = feedback_response.get("total_score", 0)
    interpretation = feedback_response.get("interpretation", "")
    feedback_summary = feedback_response.get("feedback_summary", "")

else:
    st.warning("No feedback available! Please upload a video and analyze it first.")

    scores = {
        "Content & Organization": 0,
        "Delivery & Vocal Quality": 0,
        "Body Language & Eye Contact": 0,
        "Audience Engagement": 0,
        "Language & Clarity": 0
    }

    total_score = 0
    interpretation = ""
    feedback_summary = ""

# =========================
# 计算平均分（防止除0错误）
# =========================
average_score = sum(scores.values()) / len(scores) if len(scores) > 0 else 0

# =========================
# ✅ 读取 strengths/weaknesses/suggestions
# 优先 response，其次 feedback_response（因为你终端里看到它在反馈JSON里）
# =========================
strengths = []
weaknesses = []
suggestions = []

resp_obj = st.session_state.get("response")
if isinstance(resp_obj, dict):
    strengths = resp_obj.get("strengths", []) or []
    weaknesses = resp_obj.get("weaknesses", []) or []
    suggestions = resp_obj.get("suggestions", []) or []

# fallback：如果 response 没有，就从 feedback_response 里找
if not strengths:
    strengths = feedback_response.get("strengths", []) or []
if not weaknesses:
    weaknesses = feedback_response.get("weaknesses", []) or []
if not suggestions:
    suggestions = feedback_response.get("suggestions", []) or []

# =========================
# 页面布局
# =========================
col1, col2, col3 = st.columns([0.3, 0.4, 0.3])

# =========================
# 左列：评分总结
# =========================
with col1:
    st.subheader("🧾 Evaluation Summary")
    st.markdown("<br>", unsafe_allow_html=True)

    for criterion, score in scores.items():
        label_col, progress_col, score_col = st.columns([2, 3, 1])

        with label_col:
            st.markdown(f"**{criterion}**")
        with progress_col:
            st.progress(score / 5 if score else 0)
        with score_col:
            st.markdown(f"<span><b>{score}/5</b></span>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"#### 🏆 Total Score: {total_score} / 25")
    st.markdown(f"#### 🎯 Average Score: {average_score:.2f} / 5")

    st.markdown("""---""")

    st.markdown("##### 🗣️ Feedback Summary:")
    st.markdown(f"📝 **Overall Assessment**: {interpretation}")
    st.info(f"{feedback_summary}")

# =========================
# 中列：优势劣势建议
# =========================
with col2:
    st.markdown("##### 🦾 Strengths:")
    if strengths:
        strengths_text = "\n".join(f"- {item}" for item in strengths)
        st.success(strengths_text)
    else:
        st.info("No strengths data available.")

    st.markdown("##### ⚠️ Weaknesses:")
    if weaknesses:
        weaknesses_text = "\n".join(f"- {item}" for item in weaknesses)
        st.error(weaknesses_text)
    else:
        st.info("No weaknesses data available.")

    st.markdown("##### 💡 Suggestions for Improvement:")
    if suggestions:
        suggestions_text = "\n".join(f"- {item}" for item in suggestions)
        st.warning(suggestions_text)
    else:
        st.info("No suggestions available.")

# =========================
# 右列：雷达图
# =========================
with col3:
    st.subheader("📊 Performance Chart")

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=list(scores.values()),
        theta=list(scores.keys()),
        fill="toself",
        name="Scores"
    ))

    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=False,
        margin=dict(t=50, b=50, l=50, r=50),
        width=350,
        height=350
    )

    st.plotly_chart(radar_fig, use_container_width=True)
    st.markdown("""---""")
