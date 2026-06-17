# Stage-3 reproduction image: rebuilds artifacts (offline) and runs the network-free rank step.
# Mirrors the compute constraints (CPU only, no GPU). Build then run:
#   docker build -t redrob-ranker .
#   docker run --rm -v "$PWD/out:/app/out" redrob-ranker
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4

COPY requirements.txt .
# gradio is only for the sandbox app; skip it in the reproduction image to keep it lean.
RUN sed -i '/gradio/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY ["[PUB] India_runs_data_and_ai_challenge", "./[PUB] India_runs_data_and_ai_challenge"]

# Offline pre-compute, then the reproduced ranking step. (Cross-encoder optional; omit to keep
# the image fully offline after the base build.)
CMD bash -lc "\
  python src/jd_aspects.py && \
  python src/build_index.py --embedder tfidf && \
  python src/train.py && \
  python src/rank.py --candidates '[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl' --out out/submission.csv && \
  python '[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py' out/submission.csv"
