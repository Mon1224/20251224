"""统一异常处理：不管哪一层出错，都用同一套话术回复，不把内部细节抖给用户。

对应海报里的 core/exceptions.py。
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class AppException(Exception):
    """业务异常基类。抛出它 = 主动告诉客户端"你哪里做得不对"。"""

    def __init__(self, message: str, code: str = "APP_ERROR", status_code: int = 400):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(message)


class NotFoundError(AppException):
    def __init__(self, message: str = "请求的资源不存在"):
        super().__init__(message, code="NOT_FOUND", status_code=404)


class AuthError(AppException):
    def __init__(self, message: str = "未登录或登录已过期"):
        super().__init__(message, code="UNAUTHORIZED", status_code=401)


class BusinessError(AppException):
    def __init__(self, message: str, code: str = "BUSINESS_ERROR"):
        super().__init__(message, code=code, status_code=400)


def _payload(code: str, message: str, detail=None) -> dict:
    return {"code": code, "message": message, "detail": detail}


def register_exception_handlers(app: FastAPI) -> None:
    """把处理器挂到 FastAPI 上，main.py 启动时调用一次。"""

    @app.exception_handler(AppException)
    async def _app_exception_handler(request: Request, exc: AppException):
        logger.warning("业务异常 %s %s -> %s", request.method, request.url.path, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(exc.code, exc.message),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        # 前端传参格式不对，把具体哪个字段错了告诉它
        logger.warning("参数校验失败 %s %s -> %s", request.method, request.url.path, exc.errors())
        return JSONResponse(
            status_code=422,
            content=_payload("VALIDATION_ERROR", "请求参数不合法", exc.errors()),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload("HTTP_ERROR", str(exc.detail)),
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception):
        # 兜底：任何没被预料到的错误，记详细日志，但只给客户端一句笼统的话
        logger.exception("未处理异常 %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=_payload("INTERNAL_ERROR", "服务器内部错误，请稍后再试"),
        )
