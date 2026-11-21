"""Audited paired effect resolver core used by the confirmatory family generator."""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


Interval = Tuple[float, float]


class InputError(ValueError):
    """Raised when an effect specification cannot be evaluated safely."""


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    message: str


@dataclass(frozen=True)
class Samples:
    method: Tuple[float, ...]
    baseline: Tuple[float, ...]
    design: str
    dropped_pairs: int = 0


@dataclass(frozen=True)
class EffectReport:
    metric: str
    design: str
    n_method: int
    n_baseline: int
    method_mean: float
    method_sd: Optional[float]
    baseline_mean: float
    baseline_sd: Optional[float]
    effect: float
    confidence: float
    interval_bootstrap: Optional[Interval]
    interval_t: Optional[Interval]
    interval_decision: Optional[Interval]
    threshold: float
    status: str
    pairs_at_or_above_threshold: Optional[int]
    ledger_effect: Dict[str, object]
    findings: Tuple[Finding, ...]


def _beta_fraction(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    result = d
    for m in range(1, 400):
        numerators = (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
        )
        for numerator in numerators:
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            result *= d * c
        if abs(d * c - 1.0) < 1e-13:
            break
    return result


def t_cdf(value: float, df: float) -> float:
    x = df / (df + value * value)
    a, b = df / 2.0, 0.5
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        tail = math.exp(log_front) * _beta_fraction(a, b, x) / a
    else:
        tail = 1.0 - math.exp(log_front) * _beta_fraction(b, a, 1.0 - x) / b
    return 1.0 - 0.5 * tail


def t_quantile(probability: float, df: float) -> float:
    low, high = 1e-9, 2.0
    while t_cdf(high, df) < probability and high < 1e7:
        high *= 2.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if t_cdf(middle, df) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def effect_of(method: Sequence[float], baseline: Sequence[float], sign: float) -> float:
    return sign * (statistics.fmean(method) - statistics.fmean(baseline))


def bootstrap_interval(
    samples: Samples,
    sign: float,
    confidence: float,
    resamples: int,
    seed: int,
) -> Interval:
    rng = random.Random(seed)
    method, baseline = samples.method, samples.baseline
    draws: List[float] = []
    for _ in range(resamples):
        if samples.design == "paired":
            picks = [rng.randrange(len(method)) for _ in method]
            draws.append(effect_of([method[i] for i in picks], [baseline[i] for i in picks], sign))
        else:
            new_method = [rng.choice(method) for _ in method]
            new_baseline = baseline if len(baseline) == 1 else [rng.choice(baseline) for _ in baseline]
            draws.append(effect_of(new_method, new_baseline, sign))
    draws.sort()
    alpha = (1.0 - confidence) / 2.0
    return (
        draws[max(0, math.floor(alpha * resamples))],
        draws[min(resamples - 1, math.ceil((1.0 - alpha) * resamples) - 1)],
    )


def t_interval(samples: Samples, sign: float, confidence: float) -> Interval:
    method, baseline = samples.method, samples.baseline
    centre = effect_of(method, baseline, sign)
    if samples.design == "paired" or len(baseline) == 1:
        one = [a - b for a, b in zip(method, baseline)] if samples.design == "paired" else list(method)
        error, df = statistics.stdev(one) / math.sqrt(len(one)), len(one) - 1.0
    else:
        part_m = statistics.variance(method) / len(method)
        part_b = statistics.variance(baseline) / len(baseline)
        error = math.sqrt(part_m + part_b)
        bottom = part_m ** 2 / (len(method) - 1) + part_b ** 2 / (len(baseline) - 1)
        df = (part_m + part_b) ** 2 / bottom if bottom > 0 else len(method) + len(baseline) - 2.0
    half = t_quantile(1.0 - (1.0 - confidence) / 2.0, df) * error
    return centre - half, centre + half


def classify(interval: Optional[Interval], threshold: float) -> str:
    if interval is None:
        return "unresolved"
    if interval[0] >= threshold:
        return "resolved"
    return "below_threshold" if interval[1] < threshold else "unresolved"


def _findings(
    samples: Samples,
    status: str,
    interval: Optional[Interval],
    threshold: float,
) -> Tuple[Finding, ...]:
    notes: List[Tuple[str, str, str]] = []
    fixed = samples.design == "fixed_baseline"
    smallest = len(samples.method) if fixed else min(len(samples.method), len(samples.baseline))
    if interval is None:
        notes.append(("SINGLE_RUN", "high", "one value gives no interval; the effect cannot be resolved"))
    elif status == "unresolved":
        notes.append((
            "EFFECT_UNRESOLVED",
            "high",
            f"the interval [{interval[0]:.4g}, {interval[1]:.4g}] spans the threshold {threshold:.4g}",
        ))
    elif status == "below_threshold":
        notes.append((
            "EFFECT_BELOW_MME",
            "high",
            f"the whole interval lies below the minimum meaningful effect {threshold:.4g}",
        ))
    if interval is not None and interval[1] < 0:
        notes.append((
            "EFFECT_NEGATIVE",
            "medium",
            "the method is worse than the baseline; check the positive control before reading this as a finding",
        ))
    if interval is not None and smallest < 8:
        notes.append((
            "SMALL_SAMPLE",
            "medium",
            f"{smallest} values: the bootstrap understates uncertainty, so the wider interval decides",
        ))
    if fixed:
        notes.append(("BASELINE_FIXED", "low", "the baseline is one number; its variance is not in the interval"))
    if samples.dropped_pairs:
        notes.append((
            "PAIRS_DROPPED",
            "medium",
            f"{samples.dropped_pairs} pair keys had no partner and were left out",
        ))
    return tuple(Finding(*note) for note in notes)


def resolve(
    samples: Samples,
    threshold: float,
    metric: str = "metric",
    lower_is_better: bool = False,
    confidence: float = 0.95,
    comparisons: int = 1,
    resamples: int = 10000,
    seed: int = 20261113,
) -> EffectReport:
    if threshold <= 0 or not 0.5 < confidence < 1.0 or comparisons < 1 or resamples < 100:
        raise InputError(
            "need a positive minimum meaningful effect, confidence inside the open unit interval, "
            "at least one comparison, and at least 100 resamples"
        )
    if samples.design == "paired" and len(samples.method) != len(samples.baseline):
        raise InputError("a paired design needs the same number of method and baseline values")
    if not samples.method or not samples.baseline:
        raise InputError("effect samples cannot be empty")
    values = samples.method + samples.baseline
    if any(not math.isfinite(value) for value in values):
        raise InputError("effect samples contain a nonfinite value")
    sign = -1.0 if lower_is_better else 1.0
    adjusted = 1.0 - (1.0 - confidence) / comparisons
    enough = len(samples.method) > 1 and (samples.design != "unpaired" or len(samples.baseline) > 1)
    boot = t_based = decision = None
    if enough:
        boot = bootstrap_interval(samples, sign, adjusted, resamples, seed)
        t_based = t_interval(samples, sign, adjusted)
        decision = min(boot[0], t_based[0]), max(boot[1], t_based[1])
    status = classify(decision, threshold)
    effect = effect_of(samples.method, samples.baseline, sign)
    pairs = None
    if samples.design == "paired":
        pairs = sum(1 for a, b in zip(samples.method, samples.baseline) if sign * (a - b) >= threshold)
    method_sd = statistics.stdev(samples.method) if len(samples.method) > 1 else None
    baseline_sd = statistics.stdev(samples.baseline) if len(samples.baseline) > 1 else None
    return EffectReport(
        metric=metric,
        design=samples.design,
        n_method=len(samples.method),
        n_baseline=len(samples.baseline),
        method_mean=statistics.fmean(samples.method),
        method_sd=method_sd,
        baseline_mean=statistics.fmean(samples.baseline),
        baseline_sd=baseline_sd,
        effect=effect,
        confidence=adjusted,
        interval_bootstrap=boot,
        interval_t=t_based,
        interval_decision=decision,
        threshold=threshold,
        status=status,
        pairs_at_or_above_threshold=pairs,
        ledger_effect={
            "metric": metric,
            "minimum_meaningful_effect": threshold,
            "observed": float(f"{effect:.6g}"),
            "resolved": status != "unresolved",
        },
        findings=_findings(samples, status, decision, threshold),
    )

