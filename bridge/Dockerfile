FROM python:3.11-slim

WORKDIR /app

# System deps for Chromium/Playwright
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget ca-certificates \
    libnss3 libxss1 libatk1.0-0 libatk-bridge2.0-0 \
    libdrm2 libgbm1 libxkbcommon0 libasound2 fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir "browser-use" fastapi uvicorn[standard] playwright

# Install Chromium for Playwright
RUN playwright install --with-deps chromium

COPY *.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
