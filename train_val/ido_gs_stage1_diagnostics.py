"""Independent CPU diagnostics for the 2026-09-21 Head40 IDO-GS protocol.

This module depends only on NumPy and SciPy. It does not import an existing
experiment, inspect a validation set, change labels, or alter sample weights.
``diagnose_fold`` always uses the protocol's 100 bootstrap repetitions.

``fit_beta_mixture`` and ``supervision_risk`` are also public so Stage 2 can
perform the prescribed inexpensive updates with the same numerical rules.
Numerical/model degeneracy is returned as a failed fit; malformed input raises
ValueError. A failed fit has NaN sample arrays, never silently reused weights.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, betainc, digamma, logsumexp


WRONG_RATE_CLIP = 1e-4
PARAMETER_MIN = 1e-3
PARAMETER_MAX = 1e3
MAX_EM_ITERATIONS = 200
EM_RELATIVE_TOLERANCE = 1e-6
EM_STABLE_ITERATIONS = 3
MAX_OPTIMIZER_ITERATIONS = 500
MIN_COMPONENT_FRACTION = 0.05
MIN_COMPONENT_COUNT = 30
MIN_MEAN_SEPARATION = 0.10
BOOTSTRAP_REPETITIONS = 100
BOOTSTRAP_MIN_VALID = 90
BOOTSTRAP_MIN_MEDIAN_CORRELATION = 0.90
INITIAL_COMPONENTS = (
    ((1.0, 10.0), (10.0, 1.0)),
    ((1.0, 4.0), (4.0, 1.0)),
    ((2.0, 8.0), (8.0, 2.0)),
    ((2.0, 5.0), (5.0, 2.0)),
    ((1.0, 1.0), (2.0, 1.0)),
)
_DIFFICULTIES = frozenset(("clear", "fuzzy", "reviewed_defect", "supplemental"))
_POSTERIOR_KEYS = ("pc", "pn", "cdf_clean", "cdf_noise", "epsilon")


def _json_safe(value: Any) -> Any:
    """Return only JSON-standard values, including None for nonfinite scalars."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"Unsupported diagnostics value: {type(value).__name__}")


def _rates(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("wrong_rates must be a one-dimensional array")
    if not np.all(np.isfinite(array)) or np.any((array < 0) | (array > 1)):
        raise ValueError("wrong_rates must contain finite values in [0, 1]")
    return np.clip(array, WRONG_RATE_CLIP, 1.0 - WRONG_RATE_CLIP)


def _nan_arrays(count: int) -> dict[str, np.ndarray]:
    return {key: np.full(count, np.nan, dtype=np.float64) for key in _POSTERIOR_KEYS}


def _at_parameter_boundary(parameters: np.ndarray) -> bool:
    return bool(
        np.any(parameters <= PARAMETER_MIN * (1.0 + 1e-6))
        or np.any(parameters >= PARAMETER_MAX * (1.0 - 1e-6))
    )


def _weighted_beta_mle(
    log_x: np.ndarray,
    log_one_minus_x: np.ndarray,
    weights: np.ndarray,
    initial: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    """Constrained weighted maximum likelihood, not a moments approximation."""
    total = float(np.sum(weights, dtype=np.float64))
    if not np.isfinite(total) or total <= np.finfo(np.float64).tiny:
        return None, "empty_component"
    mean_log_x = float(np.dot(weights, log_x) / total)
    mean_log_one_minus_x = float(np.dot(weights, log_one_minus_x) / total)

    def objective(log_parameters: np.ndarray) -> tuple[float, np.ndarray]:
        alpha, beta = np.exp(log_parameters)
        value = (
            betaln(alpha, beta)
            - (alpha - 1.0) * mean_log_x
            - (beta - 1.0) * mean_log_one_minus_x
        )
        shared = digamma(alpha + beta)
        gradient = np.array(
            (
                alpha * (digamma(alpha) - shared - mean_log_x),
                beta * (digamma(beta) - shared - mean_log_one_minus_x),
            ),
            dtype=np.float64,
        )
        return float(value), gradient

    bounds = [(float(np.log(PARAMETER_MIN)), float(np.log(PARAMETER_MAX)))] * 2
    try:
        result = minimize(
            objective,
            np.log(np.asarray(initial, dtype=np.float64)),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": MAX_OPTIMIZER_ITERATIONS, "ftol": 1e-12, "gtol": 1e-7},
        )
    except (FloatingPointError, OverflowError, ValueError) as exc:
        return None, f"optimizer_exception:{type(exc).__name__}"
    if not result.success:
        return None, f"optimizer_failed:{str(result.message)}"
    parameters = np.exp(result.x).astype(np.float64)
    if not np.all(np.isfinite(parameters)) or not np.isfinite(result.fun):
        return None, "optimizer_nonfinite"
    return parameters, None


def _log_joint(
    log_x: np.ndarray,
    log_one_minus_x: np.ndarray,
    parameters: np.ndarray,
    mixing: np.ndarray,
) -> np.ndarray:
    return (
        (parameters[:, 0] - 1.0) * log_x[:, None]
        + (parameters[:, 1] - 1.0) * log_one_minus_x[:, None]
        - betaln(parameters[:, 0], parameters[:, 1])
        + np.log(mixing)
    )


def _expectation(
    log_x: np.ndarray,
    log_one_minus_x: np.ndarray,
    parameters: np.ndarray,
    mixing: np.ndarray,
    multiplicities: np.ndarray,
) -> tuple[np.ndarray, float]:
    joint = _log_joint(log_x, log_one_minus_x, parameters, mixing)
    normalizer = logsumexp(joint, axis=1)
    posterior = np.exp(joint - normalizer[:, None])
    posterior /= posterior.sum(axis=1, keepdims=True)
    return posterior, float(np.dot(multiplicities, normalizer))


def _one_em_start(
    x: np.ndarray,
    multiplicities: np.ndarray,
    initial_parameters: np.ndarray,
    initial_mixing: np.ndarray,
    name: str,
) -> dict[str, Any]:
    """Fit compressed distinct rates; multiplicities preserve the exact likelihood."""
    log_x = np.log(x)
    log_one_minus_x = np.log1p(-x)
    count = float(multiplicities.sum())
    parameters = initial_parameters.copy()
    mixing = initial_mixing.copy()
    stable = 0
    converged = False
    previous_average = None
    report: dict[str, Any] = {"initialization": name, "valid": False, "failure_reasons": []}
    for iteration in range(1, MAX_EM_ITERATIONS + 1):
        posterior, old_likelihood = _expectation(
            log_x, log_one_minus_x, parameters, mixing, multiplicities
        )
        if not np.isfinite(old_likelihood) or not np.all(np.isfinite(posterior)):
            report["failure_reasons"].append("nonfinite_expectation")
            break
        weighted_posterior = multiplicities[:, None] * posterior
        effective_n = weighted_posterior.sum(axis=0)
        if np.any(effective_n <= np.finfo(np.float64).eps * count):
            report["failure_reasons"].append("empty_component")
            break
        updated = []
        for component in range(2):
            estimate, error = _weighted_beta_mle(
                log_x, log_one_minus_x, weighted_posterior[:, component], parameters[component]
            )
            if error is not None:
                report["failure_reasons"].append(f"component_{component}:{error}")
                break
            updated.append(estimate)
        if len(updated) != 2:
            break
        parameters = np.asarray(updated, dtype=np.float64)
        mixing = effective_n / count
        posterior, likelihood = _expectation(
            log_x, log_one_minus_x, parameters, mixing, multiplicities
        )
        if not np.isfinite(likelihood) or not np.all(np.isfinite(posterior)):
            report["failure_reasons"].append("nonfinite_expectation")
            break
        # A material decrease means the numerical M step did not perform valid EM.
        if likelihood < old_likelihood - 1e-8 * max(count, abs(old_likelihood)):
            report["failure_reasons"].append("log_likelihood_decreased")
            break
        average = likelihood / count
        if previous_average is not None:
            relative_change = abs(average - previous_average) / max(1.0, abs(previous_average))
            stable = stable + 1 if relative_change <= EM_RELATIVE_TOLERANCE else 0
            if stable >= EM_STABLE_ITERATIONS:
                converged = True
                break
        previous_average = average
    report["iterations"] = iteration
    report["converged"] = converged
    if not converged:
        if not report["failure_reasons"]:
            report["failure_reasons"].append("em_not_converged")
        return report

    order = np.argsort(parameters[:, 0] / parameters.sum(axis=1), kind="stable")
    parameters = parameters[order]
    mixing = mixing[order]
    posterior = posterior[:, order]
    means = parameters[:, 0] / parameters.sum(axis=1)
    effective_n = (multiplicities[:, None] * posterior).sum(axis=0)
    minimum_n = max(MIN_COMPONENT_FRACTION * count, MIN_COMPONENT_COUNT)
    report.update(
        {
            "parameters": {
                "alpha": parameters[:, 0],
                "beta": parameters[:, 1],
                "mixing": mixing,
            },
            "component_means": means,
            "mean_separation": float(means[1] - means[0]),
            "effective_n": effective_n,
            "minimum_effective_n": minimum_n,
            "log_likelihood": likelihood,
        }
    )
    if _at_parameter_boundary(parameters):
        report["failure_reasons"].append("parameter_at_numerical_boundary")
    if np.any(effective_n < minimum_n):
        report["failure_reasons"].append("insufficient_component_effective_n")
    if float(means[1] - means[0]) < MIN_MEAN_SEPARATION:
        report["failure_reasons"].append("insufficient_mean_separation")
    cdf = betainc(parameters[:, 0], parameters[:, 1], x[:, None])
    if not np.all(np.isfinite(cdf)) or np.any((cdf < 0) | (cdf > 1)):
        report["failure_reasons"].append("cdf_nonfinite_or_out_of_range")
    report["valid"] = not report["failure_reasons"]
    return report


def _parameter_arrays(initial: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    values = initial.get("parameters", initial)
    try:
        alpha = np.asarray(values["alpha"], dtype=np.float64)
        beta = np.asarray(values["beta"], dtype=np.float64)
        mixing = np.asarray(values["mixing"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("initial must supply alpha, beta and mixing arrays of length 2") from exc
    if alpha.shape != (2,) or beta.shape != (2,) or mixing.shape != (2,):
        raise ValueError("initial alpha, beta and mixing must each have shape [2]")
    parameters = np.stack((alpha, beta), axis=1)
    if (
        not np.all(np.isfinite(parameters))
        or np.any(parameters < PARAMETER_MIN)
        or np.any(parameters > PARAMETER_MAX)
        or not np.all(np.isfinite(mixing))
        or np.any(mixing <= 0)
        or not np.isclose(mixing.sum(), 1.0, atol=1e-10, rtol=0)
    ):
        raise ValueError("initial mixture parameters are outside their valid ranges")
    return parameters, mixing / mixing.sum()


def _evaluate_mixture(x: np.ndarray, parameters: Mapping[str, Any]) -> dict[str, np.ndarray]:
    beta_parameters, mixing = _parameter_arrays(parameters)
    posterior, _ = _expectation(
        np.log(x), np.log1p(-x), beta_parameters, mixing, np.ones(x.size, dtype=np.float64)
    )
    cdf_clean = betainc(beta_parameters[0, 0], beta_parameters[0, 1], x)
    cdf_noise = betainc(beta_parameters[1, 0], beta_parameters[1, 1], x)
    pc, pn = posterior[:, 0], posterior[:, 1]
    arrays = {
        "pc": pc,
        "pn": pn,
        "cdf_clean": cdf_clean,
        "cdf_noise": cdf_noise,
        "epsilon": pc * cdf_clean + pn * (1.0 - cdf_noise),
    }
    if any(
        not np.all(np.isfinite(value)) or np.any((value < 0) | (value > 1))
        for value in arrays.values()
    ) or not np.allclose(pc + pn, 1.0, rtol=0, atol=1e-12):
        raise FloatingPointError("posterior_or_cdf_out_of_range")
    return arrays


def fit_beta_mixture(
    wrong_rates: Sequence[float] | np.ndarray,
    initial: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Fit a class-wise BMM, without BIC/bootstrap or the fuzzy-defect risk gate.

    Returns ``(fit_summary, arrays)``. The summary has ``valid``,
    ``failure_reasons`` and, when fitted, ``parameters`` containing length-two
    ``alpha``, ``beta`` and ``mixing`` lists ordered clean then noise. Pass those
    parameters (or the entire summary) as ``initial`` for Stage 2. A valid warm
    start is accepted immediately; otherwise all five fixed starts are tried.
    Every returned valid fit meets the component-count and separation rules.
    """
    x = _rates(wrong_rates)
    arrays = _nan_arrays(x.size)
    summary: dict[str, Any] = {
        "valid": False,
        "n_samples": int(x.size),
        "n_distinct_clipped_rates": int(np.unique(x).size),
        "failure_reasons": [],
        "candidates": [],
    }
    parsed_initial = _parameter_arrays(initial) if initial is not None else None
    if x.size < 2 * MIN_COMPONENT_COUNT:
        summary["failure_reasons"].append("fewer_than_60_class_samples")
    if x.size == 0 or np.ptp(x) == 0:
        summary["failure_reasons"].append("constant_or_empty_input")
    if summary["failure_reasons"]:
        return _json_safe(summary), arrays
    unique_x, multiplicities = np.unique(x, return_counts=True)
    multiplicities = multiplicities.astype(np.float64)

    selected = None
    if parsed_initial is not None:
        candidate = _one_em_start(unique_x, multiplicities, *parsed_initial, "previous_parameters")
        summary["candidates"].append(candidate)
        if candidate["valid"]:
            selected = candidate
    if selected is None:
        candidates = []
        for index, start in enumerate(INITIAL_COMPONENTS):
            candidate = _one_em_start(
                unique_x,
                multiplicities,
                np.asarray(start, dtype=np.float64),
                np.array((0.5, 0.5), dtype=np.float64),
                f"fixed_{index + 1}",
            )
            summary["candidates"].append(candidate)
            if candidate["valid"]:
                candidates.append(candidate)
        if candidates:
            selected = max(candidates, key=lambda item: item["log_likelihood"])
    if selected is None:
        summary["failure_reasons"].append("no_valid_initialization")
        summary["failure_reasons"].extend(
            sorted({reason for item in summary["candidates"] for reason in item["failure_reasons"]})
        )
        return _json_safe(summary), arrays

    try:
        arrays = _evaluate_mixture(x, selected["parameters"])
    except FloatingPointError as exc:
        summary["failure_reasons"].append(str(exc))
        return _json_safe(summary), _nan_arrays(x.size)
    summary.update({key: value for key, value in selected.items() if key != "failure_reasons"})
    summary["bic"] = -2.0 * selected["log_likelihood"] + 5.0 * np.log(x.size)
    return _json_safe(summary), arrays


def _fit_single_beta(wrong_rates: np.ndarray) -> dict[str, Any]:
    x = _rates(wrong_rates)
    report: dict[str, Any] = {
        "valid": False, "converged": False, "n_samples": int(x.size), "failure_reasons": []
    }
    if x.size < 2 or np.ptp(x) == 0:
        report["failure_reasons"].append("constant_or_insufficient_input")
        return report
    unique_x, multiplicities = np.unique(x, return_counts=True)
    log_x, log_one_minus_x = np.log(unique_x), np.log1p(-unique_x)
    parameters, error = _weighted_beta_mle(
        log_x,
        log_one_minus_x,
        multiplicities.astype(np.float64),
        np.ones(2, dtype=np.float64),
    )
    if error is not None:
        report["failure_reasons"].append(error)
        return report
    alpha, beta = parameters
    likelihood = float(
        np.dot(multiplicities, (alpha - 1.0) * log_x + (beta - 1.0) * log_one_minus_x)
        - x.size * betaln(alpha, beta)
    )
    report.update(
        {
            "converged": True,
            "parameters": {"alpha": alpha, "beta": beta},
            "log_likelihood": likelihood,
            "bic": -2.0 * likelihood + 2.0 * np.log(x.size),
        }
    )
    if _at_parameter_boundary(parameters):
        report["failure_reasons"].append("parameter_at_numerical_boundary")
    if not np.isfinite(likelihood):
        report["failure_reasons"].append("nonfinite_likelihood")
    report["valid"] = not report["failure_reasons"]
    return _json_safe(report)


def _pearson(left: np.ndarray, right: np.ndarray) -> tuple[float | None, str | None]:
    if left.shape != right.shape or left.size < 2:
        return None, "correlation_insufficient_samples"
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return None, "correlation_nonfinite_input"
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return None, "correlation_constant_input"
    centered_left, centered_right = left - left.mean(), right - right.mean()
    denominator = float(np.linalg.norm(centered_left) * np.linalg.norm(centered_right))
    if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).tiny:
        return None, "correlation_undefined"
    correlation = float(np.dot(centered_left, centered_right) / denominator)
    if not np.isfinite(correlation):
        return None, "correlation_undefined"
    return float(np.clip(correlation, -1.0, 1.0)), None


def _bootstrap_stability(
    wrong_rates: np.ndarray,
    reference_pn: np.ndarray,
    seed: int,
    progress: Callable[[str], None] | None = None,
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, Any]:
    """Internal repetition override is for small numerical checks, not CLI use."""
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, np.iinfo(np.int64).max, size=repetitions, dtype=np.int64)
    records: list[dict[str, Any]] = []
    correlations: list[float] = []
    original_x = _rates(wrong_rates)
    for index, replicate_seed in enumerate(seeds):
        indices = np.random.default_rng(int(replicate_seed)).integers(0, original_x.size, size=original_x.size)
        fit, _ = fit_beta_mixture(original_x[indices])
        record: dict[str, Any] = {
            "replicate": index + 1,
            "seed": int(replicate_seed),
            "valid": False,
            "failure_reasons": list(fit["failure_reasons"]),
            "correlation": None,
            "effective_n": fit.get("effective_n"),
            "mean_separation": fit.get("mean_separation"),
        }
        if fit["valid"]:
            try:
                fitted = _evaluate_mixture(original_x, fit["parameters"])
                correlation, error = _pearson(reference_pn, fitted["pn"])
            except FloatingPointError as exc:
                correlation, error = None, str(exc)
            if error is not None:
                record["failure_reasons"].append(error)
            else:
                record["valid"] = True
                record["correlation"] = correlation
                correlations.append(correlation)
        records.append(record)
        if progress is not None and ((index + 1) % 10 == 0 or index + 1 == repetitions):
            progress(f"Bootstrap {index + 1}/{repetitions}; valid={len(correlations)}")
    median = float(np.median(correlations)) if correlations else None
    # A reduced internal test can never accidentally pass the production gate.
    gate_pass = (
        repetitions == BOOTSTRAP_REPETITIONS
        and len(correlations) >= BOOTSTRAP_MIN_VALID
        and median is not None
        and median >= BOOTSTRAP_MIN_MEDIAN_CORRELATION
    )
    reasons = []
    if repetitions != BOOTSTRAP_REPETITIONS:
        reasons.append("nonproduction_repetition_count")
    if len(correlations) < BOOTSTRAP_MIN_VALID:
        reasons.append("fewer_than_90_valid_bootstrap_fits")
    if median is None or median < BOOTSTRAP_MIN_MEDIAN_CORRELATION:
        reasons.append("median_posterior_correlation_below_0.90_or_undefined")
    return _json_safe(
        {
            "attempted": repetitions,
            "seed": seed,
            "rng": "numpy.random.default_rng(PCG64)",
            "resampling": "per-replicate default_rng(seed).integers(0, n_class, size=n_class)",
            "valid_count": len(correlations),
            "median_correlation": median,
            "minimum_valid_count": BOOTSTRAP_MIN_VALID,
            "minimum_median_correlation": BOOTSTRAP_MIN_MEDIAN_CORRELATION,
            "gate_pass": gate_pass,
            "failure_reasons": reasons,
            "replicates": records,
        }
    )


def _validate_labels_and_difficulties(
    labels: np.ndarray, difficulties: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    label_array = np.asarray(labels)
    if label_array.ndim != 1 or label_array.size == 0:
        raise ValueError("labels must be a nonempty one-dimensional array")
    if label_array.dtype.kind not in "biu" or not np.all(np.isin(label_array, (0, 1))):
        raise ValueError("labels must be integer binary labels: normal=0, defect=1")
    groups = np.asarray(difficulties)
    if groups.shape != label_array.shape:
        raise ValueError("difficulties must contain one group for each label")
    if groups.dtype.kind not in "USO" or any(value not in _DIFFICULTIES for value in groups):
        raise ValueError("difficulties must be clear/fuzzy/reviewed_defect/supplemental without missing values")
    if np.any((groups == "reviewed_defect") & (label_array != 1)):
        raise ValueError("reviewed_defect samples must have defect label 1")
    return label_array.astype(np.int64, copy=False), groups.astype(str)


def supervision_risk(
    labels: np.ndarray, difficulties: Sequence[str], pc: np.ndarray
) -> dict[str, Any]:
    """Check actual fuzzy-defect CE coefficients and report all group distributions."""
    labels, groups = _validate_labels_and_difficulties(labels, difficulties)
    coefficients = np.asarray(pc, dtype=np.float64)
    if coefficients.shape != labels.shape:
        raise ValueError("pc must have shape [N] matching labels")
    finite = np.isfinite(coefficients)
    if np.any((coefficients[finite] < 0) | (coefficients[finite] > 1)):
        raise ValueError("finite pc coefficients must be in [0, 1]")
    group_reports = {}
    for difficulty in ("clear", "fuzzy", "reviewed_defect", "supplemental"):
        for label, class_name in ((0, "normal"), (1, "defect")):
            mask = (groups == difficulty) & (labels == label)
            values = coefficients[mask]
            usable = values[np.isfinite(values)]
            group_reports[f"{difficulty}-{class_name}"] = {
                "n_samples": int(values.size),
                "n_finite": int(usable.size),
                "all_coefficients_finite": bool(values.size > 0 and usable.size == values.size),
                "mean_pc": float(usable.mean()) if usable.size else None,
                "median_pc": float(np.median(usable)) if usable.size else None,
                "min_pc": float(usable.min()) if usable.size else None,
                "max_pc": float(usable.max()) if usable.size else None,
                "pc_quantiles_10_25_50_75_90": np.quantile(usable, (0.1, 0.25, 0.5, 0.75, 0.9)) if usable.size else None,
                "fraction_pc_below_0.5": float(np.mean(usable < 0.5)) if usable.size else None,
            }
    target = coefficients[(groups == "fuzzy") & (labels == 1)]
    reasons = []
    complete = bool(target.size and np.all(np.isfinite(target)))
    r_value = float(target.mean()) if complete else None
    q_value = float(np.mean(target < 0.5)) if complete else None
    if target.size == 0:
        reasons.append("data_protocol_incomplete:empty_fuzzy_defect_group")
    elif not complete:
        reasons.append("fuzzy_defect_pc_unavailable_or_nonfinite")
    else:
        if r_value < 0.80:
            reasons.append("fuzzy_defect_mean_pc_below_0.80")
        if q_value > 0.10:
            reasons.append("fuzzy_defect_fraction_pc_below_0.5_exceeds_0.10")
    return _json_safe(
        {
            "gate_pass": not reasons,
            "failure_reasons": reasons,
            "n_fuzzy_defect": int(target.size),
            "R_fuzzy_defect": r_value,
            "Q_fuzzy_defect": q_value,
            "minimum_mean_pc": 0.80,
            "maximum_fraction_pc_below_0.5": 0.10,
            "groups": group_reports,
        }
    )


def diagnose_fold(
    predictions: np.ndarray,
    labels: np.ndarray,
    difficulties: Sequence[str],
    seed: int,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Diagnose 45 ordered training predictions (epochs 6..50), shape [45, N].

    Caller must ensure columns keep a stable SampleID order and are training
    samples only. ``summary`` is serializable with ``json.dumps(allow_nan=False)``.
    Half windows and PC are diagnostic-only; they never affect ``gate_pass``.
    Bootstrap is skipped when a full-window structural gate has already failed.
    The skipped check is always a failed admission condition, never a pass.
    """
    labels, groups = _validate_labels_and_difficulties(labels, difficulties)
    predictions = np.asarray(predictions)
    if predictions.shape != (45, labels.size):
        raise ValueError("predictions must have shape [45, N], ordered epochs 6..50")
    if predictions.dtype.kind not in "biu" or not np.all(np.isin(predictions, (0, 1))):
        raise ValueError("predictions must contain integer binary predictions")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    wrong = predictions != labels[None, :]
    changes = predictions[1:] != predictions[:-1]
    wrong_count = wrong.sum(axis=0, dtype=np.int64)
    wrong_rate = wrong_count.astype(np.float64) / 45.0
    arrays = {
        "wrong_count": wrong_count,
        "wrong_rate": wrong_rate,
        "flip_rate": changes.mean(axis=0, dtype=np.float64),
        **_nan_arrays(labels.size),
    }
    pc_curve = changes.mean(axis=1, dtype=np.float64)
    summary: dict[str, Any] = {
        "protocol": "Head40_IDO_GS_2026-09-21",
        "seed": int(seed),
        "n_samples": int(labels.size),
        "n_observed": 45,
        "prediction_epochs": [6, 50],
        "classwise": {},
        "PC": {
            "epochs": list(range(7, 51)),
            "values": pc_curve,
            "early_epochs": [7, 10],
            "early_mean": float(pc_curve[:4].mean()),
            "late_epochs": [45, 50],
            "late_mean": float(pc_curve[-6:].mean()),
            "diagnostic_only": True,
        },
        "failure_reasons": [],
    }
    for class_id, class_name in ((0, "normal"), (1, "defect")):
        mask = labels == class_id
        if progress is not None:
            progress(f"{class_name}: full-window Beta/BMM, n={int(mask.sum())}")
        rates = wrong_rate[mask]
        single = _fit_single_beta(rates)
        mixture, fitted_arrays = fit_beta_mixture(rates)
        for key in _POSTERIOR_KEYS:
            arrays[key][mask] = fitted_arrays[key]
        reasons = []
        if not single["valid"]:
            reasons.extend(f"single_beta:{value}" for value in single["failure_reasons"])
        if not mixture["valid"]:
            reasons.extend(f"mixture:{value}" for value in mixture["failure_reasons"])
        bic_improvement = None
        if single["valid"] and mixture["valid"]:
            bic_improvement = single["bic"] - mixture["bic"]
            if not mixture["bic"] < single["bic"] - 10.0:
                reasons.append("BIC_2_not_less_than_BIC_1_minus_10")
        structure = {
            "gate_pass": not reasons,
            "failure_reasons": reasons,
            "bic_improvement": bic_improvement,
            "required_bic_improvement_strictly_greater_than": 10.0,
        }
        if structure["gate_pass"]:
            bootstrap = _bootstrap_stability(
                rates,
                fitted_arrays["pn"],
                int(seed) + 10000 + class_id,
                (lambda message, name=class_name: progress(f"{name}: {message}")) if progress else None,
            )
        else:
            bootstrap = {
                "attempted": 0,
                "planned_repetitions": BOOTSTRAP_REPETITIONS,
                "valid_count": 0,
                "median_correlation": None,
                "seed": int(seed) + 10000 + class_id,
                "gate_pass": False,
                "failure_reasons": ["skipped_due_to_structure_failure"],
                "replicates": [],
            }
        half_windows = {}
        for name, window, epochs in (
            ("early", slice(0, 22), [6, 27]),
            ("late", slice(22, 45), [28, 50]),
        ):
            half_rates = wrong[window][:, mask].mean(axis=0, dtype=np.float64)
            half_fit, half_arrays = fit_beta_mixture(half_rates)
            if half_fit["valid"] and mixture["valid"]:
                correlation, error = _pearson(fitted_arrays["pn"], half_arrays["pn"])
            else:
                correlation, error = None, "full_or_half_window_mixture_invalid"
            half_windows[name] = {
                "epochs": epochs,
                "n_observed": epochs[1] - epochs[0] + 1,
                "mixture": half_fit,
                "posterior_correlation_with_full": correlation,
                "correlation_failure_reason": error,
                "diagnostic_only": True,
            }
        class_pass = structure["gate_pass"] and bootstrap["gate_pass"]
        summary["classwise"][class_name] = {
            "class_id": class_id,
            "n_samples": int(mask.sum()),
            "single_beta": single,
            "mixture": mixture,
            "structure_gate": structure,
            "bootstrap": bootstrap,
            "half_windows": half_windows,
            "gate_pass": class_pass,
        }
        summary["failure_reasons"].extend(f"{class_name}:{reason}" for reason in reasons)
        summary["failure_reasons"].extend(
            f"{class_name}:{reason}" for reason in bootstrap["failure_reasons"]
        )
    summary["risk"] = supervision_risk(labels, groups, arrays["pc"])
    summary["failure_reasons"].extend(f"risk:{reason}" for reason in summary["risk"]["failure_reasons"])
    summary["gate_pass"] = not summary["failure_reasons"]
    summary["conclusion"] = "passed_this_fold_admission_conditions" if summary["gate_pass"] else "did_not_pass_admission_conditions"
    return _json_safe(summary), arrays


__all__ = ["diagnose_fold", "fit_beta_mixture", "supervision_risk"]
