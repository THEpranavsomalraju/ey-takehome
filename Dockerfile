# Optional guaranteed environment. The primary path remains a plain virtual environment.
#   docker build -t ledgerclf .
#   docker run --rm ledgerclf                      (runs the doctor)
#   docker run --rm -p 8000:8000 ledgerclf python -m ledgerclf serve --host 0.0.0.0
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m pip install --no-cache-dir -e . --no-deps

CMD ["python", "-m", "ledgerclf", "doctor"]
