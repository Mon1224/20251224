"""Smoke tests for the AI Speech Trainer project.

Run from the project root:
    python -m unittest test_smoke -v

The backend tests import the real agent stack (mediapipe, tensorflow, agno),
so the first run can take a minute.
"""

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"


class VoiceToolTests(unittest.TestCase):
    """The voice analysis tool must fail explicitly, never crash."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(BACKEND))
        from agents.tools.voice_analysis_tool import analyze_voice_attributes
        cls.tool = analyze_voice_attributes

    def test_missing_file_returns_error_json(self):
        out = self.tool.entrypoint("C:/does/not/exist/video.mp4")
        data = json.loads(out)
        self.assertTrue(data.get("error"))
        self.assertEqual(data.get("transcription"), "")
        self.assertEqual(data.get("speech_rate_wpm"), "N/A")


class FacialToolTests(unittest.TestCase):
    """The facial analysis tool must fail explicitly, never crash."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(BACKEND))
        from agents.tools.facial_expression_tool import analyze_facial_expressions
        cls.tool = analyze_facial_expressions

    def test_missing_video_returns_error_json(self):
        out = self.tool.entrypoint("C:/does/not/exist/video.mp4")
        data = json.loads(out)
        self.assertTrue(data.get("error"))
        self.assertEqual(data.get("emotion_timeline"), [])


class FrontendPageTests(unittest.TestCase):
    """Both Streamlit pages must run without exceptions."""

    def _run_page(self, page_name: str):
        from streamlit.testing.v1 import AppTest

        # Make the frontend directory importable (os.chdir alone does not update
        # an already-resolved sys.path).
        sys.path.insert(0, str(FRONTEND))
        cwd = os.getcwd()
        os.chdir(FRONTEND)
        try:
            at = AppTest.from_file(str(FRONTEND / page_name), default_timeout=180)
            at.run()
            self.assertFalse(at.exception, msg=str(at.exception))
        finally:
            os.chdir(cwd)

    def test_home_page_runs(self):
        self._run_page("Home.py")

    def test_feedback_page_runs(self):
        self._run_page(Path("pages") / "1 - Feedback.py")


try:
    from fastapi.testclient import TestClient
    _HAS_TEST_CLIENT = True
except ImportError:
    _HAS_TEST_CLIENT = False


@unittest.skipUnless(_HAS_TEST_CLIENT, "fastapi TestClient requires httpx")
class BackendApiTests(unittest.TestCase):
    """Backend endpoints return proper error codes for bad input."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(BACKEND))
        import main
        cls.client = TestClient(main.app)

    def test_root_endpoint(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)

    def test_analyze_empty_url_returns_400(self):
        r = self.client.post("/analyze", json={"video_url": ""})
        self.assertEqual(r.status_code, 400)

    def test_analyze_missing_file_returns_404(self):
        r = self.client.post("/analyze", json={"video_url": "C:/does/not/exist/video.mp4"})
        self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
