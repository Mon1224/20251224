from abc import ABC, abstractmethod

class FacialExpressionAnalyzer(ABC):
    """
    表情分析能力的统一接口
    """

    @abstractmethod
    def analyze(self, video_path: str) -> str:
        """
        分析视频中的面部表情，并返回文字描述
        """
        pass