import streamlit as st
import requests
import tempfile
import os
import numpy as np
import json
from page_config import render_page_config

render_page_config()

# =========================
# Initialize session state
# =========================
if "begin" not in st.session_state:
    st.session_state.begin = False

if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "upload_file" not in st.session_state:
    st.session_state.upload_file = False

if "response" not in st.session_state:
    st.session_state.response = None

# ⭐ 新增：初始化 feedback_response
if "feedback_response" not in st.session_state:
    st.session_state.feedback_response = None


def clear_session_response():
    st.session_state.response = None
    st.session_state.feedback_response = None


# =========================
# Layout
# =========================
col1, col2 = st.columns([0.7, 0.3])

# =========================
# Left column
# =========================
with col1:

    spacer1, btn_col = st.columns([0.8, 0.2])

    if st.session_state.begin:
        with spacer1:
            st.markdown("<h4>📽️ Video</h4>", unsafe_allow_html=True)

        with btn_col:
            if st.button("📤 Upload Video"):
                if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                    os.remove(st.session_state.video_path)

                st.session_state.video_path = None
                clear_session_response()
                st.session_state.upload_file = True
                st.rerun()

    # Upload logic
    if st.session_state.get("upload_file"):
        uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4"])

        if uploaded_file is not None:
            temp_dir = tempfile.gettempdir()
            unique_name = f"{int(np.random.rand() * 1e8)}_{uploaded_file.name}"
            file_path = os.path.join(temp_dir, unique_name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            st.session_state.video_path = file_path
            st.session_state.upload_file = False
            st.rerun()

    # Welcome page
    if not st.session_state.begin:
        st.success("""
            **Welcome to AI Speech Trainer!**  
            Your ultimate companion to help improve your public speaking skills.
            """)
        st.info("""
                🚀 To get started:
                \n\t1. Record a speech video.
                \n\t2. Upload it.
                \n\t3. Click Analyze to get AI feedback.
                """)

        if st.button("👉 Let's begin!"):
            st.session_state.begin = True
            st.rerun()

    # Video preview + analyze
    if st.session_state.video_path:
        st.video(st.session_state.video_path, autoplay=False)

        if st.button("▶️ Analyze Video"):
            with st.spinner("Analyzing video... please wait ⏳"):

                API_URL = "http://localhost:8000/analyze"

                try:
                    response = requests.post(
                        API_URL,
                        json={"video_url": st.session_state.video_path},
                        timeout=300
                    )
                except Exception as e:
                    st.error(f"❌ Backend connection failed: {e}")
                    st.stop()

                if response.status_code == 200:
                    try:
                        data = response.json()
                    except Exception:
                        st.error("❌ Backend did not return valid JSON.")
                        st.code(response.text)
                        st.stop()

                    if not isinstance(data, dict):
                        st.error("❌ Backend response format invalid.")
                        st.code(data)
                        st.stop()

                    # -------------------------
                    # 从 backend 直接取数据
                    # -------------------------
                    trimmed_timeline = (data.get("emotion_timeline") or [])[:100]

                    # ✅ 保存整包数据，Feedback 页面会用到
                    st.session_state.response = {
                        "engagement_metrics": data.get("engagement_metrics", {}) or {},
                        "emotion_timeline": trimmed_timeline,
                        "strengths": data.get("strengths", []) or [],
                        "weaknesses": data.get("weaknesses", []) or [],
                        "suggestions": data.get("suggestions", []) or [],
                        # ✅ 兜底：把 feedback_response 也放进去（就算 session_state.feedback_response 丢了也能从这里读）
                        "feedback_response": data.get("feedback_response", "{}"),
                    }

                    # ⭐ 关键：直接用 LLM 的反馈（给 Feedback.py 的主入口）
                    st.session_state.feedback_response = data.get("feedback_response", "{}")

                    st.success("✅ Video analysis completed!")
                else:
                    st.error("🚨 Error during video analysis.")

# =========================
# Right column
# =========================
with col2:

    st.markdown("<h4>📊 Analysis Result</h4>", unsafe_allow_html=True)

    if st.session_state.response:

        data = st.session_state.response

        st.subheader("😊 Engagement Metrics")

        metrics = data.get("engagement_metrics", {})

        if metrics:
            smile = metrics.get("smile_frequency", 0)
            eye = metrics.get("eye_contact_frequency", 0)

            st.write(f"🙂 Smile Frequency: {smile}")
            st.write(f"👁️ Eye Contact Frequency: {eye}")
        else:
            st.write("No engagement metrics available.")

        st.subheader("📈 Emotion Timeline (Preview)")

        timeline = data.get("emotion_timeline", [])

        if timeline:
            st.write(f"Total Frames Stored: {len(timeline)} (max 100)")
            st.json(timeline[:20])
        else:
            st.write("No emotion data available.")

        if st.button("📝 Get Feedback"):
            st.switch_page("pages/1 - Feedback.py")

    else:
        st.markdown(
            """
            <div style="background-color:#f0f2f6; padding: 1.5rem;
                        border-radius: 10px; border: 1px solid #ccc;
                        font-family: 'Segoe UI', sans-serif;
                        line-height: 1.6; color: #333;
                        height: 400px; max-height: 400px;
                        overflow-y: auto;">
                Your analysis result will appear here.
            </div>
            <br>
            """,
            unsafe_allow_html=True
        )
