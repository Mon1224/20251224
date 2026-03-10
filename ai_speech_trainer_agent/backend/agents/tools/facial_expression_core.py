import requests

# 外部 CV 服务地址
CV_SERVICE_URL = "http://127.0.0.1:9001/analyze"


def analyze_facial_expressions_core(video_path: str) -> dict:
    """
    调用外部 CV 服务进行面部表情分析。
    本函数只负责：
        1. 发送 HTTP 请求
        2. 获取原始视觉分析结果
    不负责：
        - 任何 LLM 解释
        - 任何额外加工
    """
    print(">>> CORE RECEIVED PATH:", repr(video_path))
    print(">>> TYPE:", type(video_path))
    print(">>> SENDING JSON:", {"video_path": video_path})
    print(">>> CV_SERVICE_URL:", CV_SERVICE_URL)
    resp = requests.post(
        CV_SERVICE_URL,
        json={"video_path": video_path},
        timeout=300
    )

    if resp.status_code != 200:
        raise RuntimeError(f"CV service error: {resp.text}")

    # 直接返回 CV 结果（纯客观数据）
    return resp.json()
