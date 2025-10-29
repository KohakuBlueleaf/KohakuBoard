FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

COPY ./pyproject.toml .
RUN mkdir -p /app/src/kohakuboard
RUN echo "" > /app/src/kohakuboard/__init__.py
RUN uv pip install --system -e .

COPY ./src/kohakuboard ./src/kohakuboard

EXPOSE 48889
CMD ["uvicorn", "kohakuboard.main:app", "--host", "0.0.0.0", "--port", "48889", "--workers", "4"]
