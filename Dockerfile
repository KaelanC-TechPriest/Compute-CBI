FROM docker.io/python:3.12-slim

# Install system dependencies for GDAL / geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy uv files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy the rest of the code
COPY . .

# Pass the script path (relative to /app) and any args as CMD:
#   docker run <image> scripts/cbi_oneshot.py --mtbs-id MT3456789 ...
ENTRYPOINT ["uv", "run"]
CMD ["--help"]
