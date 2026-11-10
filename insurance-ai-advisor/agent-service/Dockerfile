FROM python:3.11-slim

# 直接用 uv 官方镜像里的二进制，不在容器里重新装一遍
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# 先装依赖再拷代码：改业务代码不会导致依赖层缓存失效
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8000

# 容器里用 0.0.0.0 监听，否则只有容器内部能访问
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
