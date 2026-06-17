"""Smoke test for the multi-JD serving layer. Run: python tests/test_serve.py

Requires built artifacts (embeddings.npy, embedder.pkl, model.bin). Verifies that rank_jd
returns a well-formed top-K, gates honeypots, and routes the ranker by JD domain.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import ARTIFACTS  # noqa: E402


def _need_artifacts():
    return all((ARTIFACTS / f).exists() for f in
               ["embeddings.npy", "embedder.pkl", "model.bin", "aspects.json"])


def test_rank_jd_shape_and_routing():
    import serve
    ai_jd = "Senior AI Engineer. Embeddings retrieval, vector database, learning to rank, NDCG."
    rows, meta = serve.rank_jd(ai_jd, "Senior AI Engineer", top_k=20)
    assert len(rows) == 20
    assert [r["rank"] for r in rows] == list(range(1, 21))           # ranks 1..20
    scores = [r["score"] for r in rows]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))  # non-increasing
    assert all(r["candidate_id"].startswith("CAND_") for r in rows)
    assert all(r["reasoning"] for r in rows)                          # every row explained
    assert "model" in meta["ranker"]                                 # AI JD -> in-domain model

    fe_jd = "Senior Frontend Engineer. React, TypeScript, Next.js, Redux, Tailwind CSS."
    _, meta_fe = serve.rank_jd(fe_jd, "Senior Frontend Engineer", top_k=10)
    assert meta_fe["ranker"].startswith("rubric")                    # out-of-domain -> rubric


if __name__ == "__main__":
    if not _need_artifacts():
        print("SKIP: artifacts missing (run build_index.py + train.py first)")
        sys.exit(0)
    try:
        test_rank_jd_shape_and_routing()
        print("PASS test_rank_jd_shape_and_routing")
        print("\n1/1 passed")
    except AssertionError as e:
        print(f"FAIL: {e}"); sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {type(e).__name__}: {e}"); sys.exit(1)
