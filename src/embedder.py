"""Pluggable text embedder used ONLY at offline build time (build_index.py).

Primary: sentence-transformers `all-MiniLM-L6-v2` (384-d, compact, CPU-friendly). The model
downloads once from HuggingFace (network is allowed during pre-compute) and is then cached;
the rank step never needs it because JD aspect vectors are precomputed into artifacts/.

Fallback: TF-IDF + TruncatedSVD (LSA), fully offline and dependency-light, used only if
sentence-transformers/torch are unavailable. Both paths emit L2-normalized float32 vectors
so cosine similarity is a plain dot product.
"""
from __future__ import annotations

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _l2norm(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


class STEmbedder:
    """sentence-transformers backend."""

    kind = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_NAME)
        self.dim = self.model.get_sentence_embedding_dimension()

    def fit(self, texts: list[str]) -> None:  # no-op; pretrained
        pass

    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        vecs = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return vecs.astype(np.float32)


class TfidfSVDEmbedder:
    """Offline fallback: TF-IDF -> TruncatedSVD (LSA)."""

    kind = "tfidf-svd"

    def __init__(self, dim: int = 256) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.dim = dim
        self.vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                   sublinear_tf=True, min_df=3)
        self.svd = TruncatedSVD(n_components=dim, random_state=42)
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        X = self.vec.fit_transform(texts)
        self.svd.fit(X)
        self._fitted = True

    def encode(self, texts: list[str], batch_size: int = 4096) -> np.ndarray:
        assert self._fitted, "TfidfSVDEmbedder must be fit() on the corpus first"
        X = self.vec.transform(texts)
        return _l2norm(self.svd.transform(X).astype(np.float32))


def save_embedder(emb, path) -> None:
    """Persist a fitted embedder so arbitrary JD queries can be embedded at serve time."""
    import pickle

    if isinstance(emb, TfidfSVDEmbedder):
        payload = {"kind": "tfidf", "vec": emb.vec, "svd": emb.svd, "dim": emb.dim}
    else:  # ST is reconstructible from the model name (weights are cached on disk)
        payload = {"kind": "st", "model": MODEL_NAME}
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)


def load_embedder(path):
    """Load an embedder saved by save_embedder()."""
    import pickle

    with open(path, "rb") as fh:
        d = pickle.load(fh)
    if d["kind"] == "tfidf":
        e = TfidfSVDEmbedder(dim=d["dim"])
        e.vec, e.svd, e._fitted = d["vec"], d["svd"], True
        return e
    return STEmbedder()


def get_embedder(prefer: str = "auto"):
    """Return an embedder.

    prefer="st"     -> sentence-transformers (raises if unavailable)
    prefer="tfidf"  -> TF-IDF+SVD (fast, fully offline, no torch)
    prefer="auto"   -> ST if importable, else TF-IDF fallback
    """
    if prefer == "tfidf":
        return TfidfSVDEmbedder()
    if prefer == "st":
        return STEmbedder()
    try:
        return STEmbedder()
    except Exception as e:  # noqa: BLE001
        print(f"[embedder] sentence-transformers unavailable ({e}); using TF-IDF+SVD fallback")
        return TfidfSVDEmbedder()
