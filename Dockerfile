FROM python:3.12-slim

WORKDIR /app

# uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 의존성 먼저 복사 (캐시 활용)
COPY pyproject.toml .
RUN uv pip install --system -e .

COPY src/ src/

EXPOSE 8000
CMD ["uvicorn", "src.agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
