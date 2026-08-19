import streamlit as st
from sidebar import render_sidebar
import os


def render_page_config():
    # Set page configuration
    st.set_page_config(
        page_icon="🎙️",
        page_title="AI Speech Trainer",
        initial_sidebar_state="auto",
        layout="wide")

    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(current_dir, "style.css")

    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Sidebar
    render_sidebar()

    # Main title with an icon
    st.markdown(
        """
        <div class="custom-header"'>
            <span>🗣️ AI Speech Trainer</span><br>
            <span>Your personal coach for public speaking</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Horizontal line
    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)