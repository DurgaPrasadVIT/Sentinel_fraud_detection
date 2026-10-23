# One image for every Sentinel service (API, case service, ring detector,
# dashboard, simulator); docker-compose.yml picks the command for each.
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first: this layer is reused until requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-deps .

# Don't run as root. Folders that get mounted must exist here, owned by this user:
# a new Docker volume copies the ownership of the folder it's mounted on. Without
# this, Docker creates /app/cases as root and the services can't write to it.
RUN useradd --create-home --uid 1000 sentinel \
    && mkdir -p /app/data /app/models /app/cases \
    && chown -R sentinel:sentinel /app
USER sentinel

# The model is NOT baked into the image: models/ is mounted at run time, so a new
# model can be deployed without rebuilding the code (and the image stays small).
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# One process per container; scale by running more containers (replicas).
# Also avoids a ~40 ms delay per request: with --workers > 1, uvicorn creates the
# listening socket without proto=IPPROTO_TCP, so asyncio never enables TCP_NODELAY,
# and on reused connections Nagle's algorithm + delayed ACK add ~40 ms (docs/PERFORMANCE.md).
CMD ["uvicorn", "sentinel.api:app", "--host", "0.0.0.0", "--port", "8000"]
