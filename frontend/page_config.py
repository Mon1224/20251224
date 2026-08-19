import streamlit as st
from pathlib import Path
from sidebar import render_sidebar

STYLE_CSS_PATH = Path(__file__).resolve().parent / "style.css"


def render_page_config():
    # Set page configuration
    st.set_page_config(
        page_icon="🎙️",
        page_title="AI Speech Trainer",
        initial_sidebar_state="auto",
        layout="wide")

    # Load external CSS. The path is resolved from this file's location, so it
    # works no matter which directory Streamlit is started from.
    with open(STYLE_CSS_PATH, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Sidebar
    render_sidebar()

    # Main title with an icon
    st.markdown(
        """
        <div class="custom-header">
            <span>🗣️ AI Speech Trainer</span><br>
            <span>Your personal coach for public speaking</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Horizontal line
    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)
