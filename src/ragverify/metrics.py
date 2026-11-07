"""Answer matching, selective-prediction metrics, statistics and risk control.

match_selfrag reproduces Self-RAG's `match` exactly (retrieval_lm/metrics.py, copied in
literature/code_refs/): case-sensitive raw substring `gt in prediction`. normalize/em/f1 follow the
SQuAD evaluation script (also used in Self-RAG's metrics.py normalize_answer).
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter

import numpy as np
from scipy import stats


def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def match_selfrag(pred: str, golds: list[str]) -> int:
    return int(any(g in pred for g in golds))


def match_norm(pred: str, golds: list[str]) -> int:
    p = normalize_answer(pred)
    return int(any(normalize_answer(g) and normalize_answer(g) in p for g in golds))


def em(pred: str, golds: list[str]) -> int:
    p = normalize_answer(pred)
    return int(any(p == normalize_answer(g) for g in golds))


def f1(pred: str, golds: list[str]) -> float:
    pt = normalize_answer(pred).split()
    best = 0.0
    for g in golds:
        gt = normalize_answer(g).split()
        common = Counter(pt) & Counter(gt)
        same = sum(common.values())
        if same == 0:
            continue
        pr, rc = same / len(pt), same / len(gt)
        best = max(best, 2 * pr * rc / (pr + rc))
    return best


# ---------------- selective prediction ----------------
def selective(correct: np.ndarray, answered: np.ndarray) -> dict:
    correct, answered = np.asarray(correct, float), np.asarray(answered, bool)
    n = len(correct)
    na = int(answered.sum())
    wrong_answered = int(((1 - correct) * answered).sum())
    return {"n": n, "coverage": na / n, "abstention_rate": 1 - na / n,
            "selective_accuracy": float(correct[answered].mean()) if na else float("nan"),
            "hallucination_rate": wrong_answered / n, "accuracy_all": float((correct * answered).sum() / n)}


def risk_coverage(scores: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Sort by descending confidence (ties broken by index); returns coverage, selective risk, AURC."""
    scores, correct = np.asarray(scores, float), np.asarray(correct, float)
    order = np.lexsort((np.arange(len(scores)), -scores))
    err = 1 - correct[order]
    k = np.arange(1, len(err) + 1)
    risk = np.cumsum(err) / k
    return k / len(err), risk, float(risk.mean())


def at_coverage(scores: np.ndarray, correct: np.ndarray, cov: float) -> dict:
    """Answer the top ceil(cov*N) by score; hallucination rate = wrong answered / N."""
    scores, correct = np.asarray(scores, float), np.asarray(correct, float)
    n = len(scores)
    m = int(math.ceil(cov * n))
    order = np.lexsort((np.arange(n), -scores))
    ans = np.zeros(n, bool)
    ans[order[:m]] = True
    return selective(correct, ans)


# ---------------- statistics ----------------
def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict:
    """CI for mean(a) - mean(b) on paired per-example values."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return {"diff": float(d.mean()), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "n": int(len(d))}


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, int), np.asarray(b, int)
    n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5)))
    return {"b_only": n01, "a_only": n10, "p": p}


def paired_permutation(a: np.ndarray, b: np.ndarray, n_perm: int = 10_000, seed: int = 0) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    return float((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).mean())


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda x: x[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


# ---------------- risk control (M6) ----------------
def crc_threshold(scores: np.ndarray, wrong: np.ndarray, alpha: float, B: float = 1.0) -> float:
    """Conformal risk control for L(lam) = 1{score >= lam} * wrong (non-increasing in lam).
    Returns the smallest lam with n/(n+1) * Rhat(lam) + B/(n+1) <= alpha (abstain if score < lam);
    +inf if even answering nothing does not certify (only possible when B/(n+1) > alpha)."""
    s, w = np.asarray(scores, float), np.asarray(wrong, float)
    n = len(s)
    if B / (n + 1) > alpha:
        return float("inf")
    cands = np.concatenate([np.unique(s), [np.inf]])
    for lam in cands:  # ascending: first feasible lam is the infimum
        rhat = float(((s >= lam) * w).mean())
        if n / (n + 1) * rhat + B / (n + 1) <= alpha:
            return float(lam)
    return float("inf")


def ltt_threshold(scores: np.ndarray, wrong: np.ndarray, alpha: float, delta: float = 0.1, grid: int = 200) -> float:
    """Learn-then-Test with fixed-sequence testing for the error rate among answered questions.
    H0(lam): P(wrong | answered at lam) > alpha; binomial p-value; test lam from strict to lenient."""
    s, w = np.asarray(scores, float), np.asarray(wrong, int)
    lams = np.unique(np.quantile(s, np.linspace(1.0, 0.0, grid)))[::-1]
    chosen = float("inf")
    for lam in lams:
        ans = s >= lam
        n, k = int(ans.sum()), int(w[ans].sum())
        # Skip thresholds where H0 cannot be rejected even with zero errors ((1-alpha)^n > delta). Whether to skip depends
        # on the scores only (not on labels), so the fixed-sequence order is still fixed given the scores. Without this,
        # the strictest threshold (n = 1) always fails and stops the sequence (bug found 2026-11-06; PROTOCOL §17).
        if n == 0 or (1 - alpha) ** n > delta:
            continue
        p = float(stats.binom.cdf(k, n, alpha))
        if p <= delta:
            chosen = float(lam)
        else:
            break
    return chosen
