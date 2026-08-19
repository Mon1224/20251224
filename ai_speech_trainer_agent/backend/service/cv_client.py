import requests

CV_SERVICE_URL = "http://127.0.0.1:9001"

def analyze_facial_expressions(video_path: str) -> dict:
    resp = requests.post(
        f"{CV_SERVICE_URL}/analyze",
        json={"video_path": video_path},
        timeout=300
    )
    resp.raise_for_status()
    return resp.json()