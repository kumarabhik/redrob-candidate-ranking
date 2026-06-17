"""Unit tests for recall primitives (RRF, tokenize). Run: python tests/test_recall.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recall import rrf, tokenize  # noqa: E402


def test_tokenize_basic():
    assert tokenize("React, TypeScript & Node.js!") == ["react", "typescript", "node.js"]


def test_rrf_orders_by_consensus():
    # item 2 is top of both lists -> must win; item present in both beats item in one.
    a = np.array([2, 1, 0])
    b = np.array([2, 3, 1])
    fused = rrf([a, b], k=60).tolist()
    assert fused[0] == 2                       # consensus winner
    assert set([1, 2, 3]).issubset(set(fused))  # union preserved


def test_rrf_truncates():
    a = np.arange(100)
    fused = rrf([a], k=60, n=10)
    assert len(fused) == 10
    assert fused[0] == 0


def test_rrf_deterministic():
    a, b = np.array([5, 4, 3]), np.array([3, 4, 5])
    assert rrf([a, b]).tolist() == rrf([a, b]).tolist()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
