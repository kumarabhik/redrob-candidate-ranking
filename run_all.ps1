# Full reproduce pipeline (Windows PowerShell). See README.md / agents.md §2.
# Offline pre-compute (network allowed) -> then the network-free ranking step.
$ErrorActionPreference = "Stop"
$data = ".\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
$validator = ".\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\validate_submission.py"

Write-Host "== [1/5] JD aspects ==" -ForegroundColor Cyan
python src\jd_aspects.py

Write-Host "== [2/5] build index (embeddings + JD vectors) ==" -ForegroundColor Cyan
python src\build_index.py --embedder tfidf        # use 'st' for MiniLM/BGE (slower, offline)

Write-Host "== [3/5] (optional) cross-encoder teacher ==" -ForegroundColor Cyan
python src\cross_encode.py --top 600              # comment out to skip; rank works without it

Write-Host "== [4/5] train LambdaMART ==" -ForegroundColor Cyan
python src\train.py

Write-Host "== [5/5] RANK STEP (<=5 min, CPU, no network) ==" -ForegroundColor Cyan
python src\rank.py --candidates $data --out .\submission.csv

Write-Host "== validate ==" -ForegroundColor Cyan
python $validator .\submission.csv
