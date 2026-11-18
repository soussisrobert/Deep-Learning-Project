"""CPU-only OPE metrics with explicit invalid-frame and aggregation contracts.

Success follows the pinned HiFT toolkit, not trapezoidal integration: strict
IoU > np.arange(0, 1.05, .05), then arithmetic mean of 21 ordinates.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

SUCCESS_THRESHOLDS = np.arange(0, 1.05, 0.05)
PRECISION_THRESHOLDS = np.arange(51, dtype=float)
NORMALIZED_THRESHOLDS = np.arange(51, dtype=float) / 100


class ValidationError(ValueError):
    """An input cannot be scored without silently changing the protocol."""


def boxes_array(value, label="boxes") -> np.ndarray:
    try:
        boxes = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label}: expected numeric Nx4 xywh rows") from exc
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not len(boxes):
        raise ValidationError(f"{label}: expected nonempty Nx4 xywh rows, got {boxes.shape}")
    return boxes


def valid_boxes(boxes: np.ndarray) -> np.ndarray:
    return np.isfinite(boxes).all(axis=1) & (boxes[:, 2:] > 0).all(axis=1)


def evaluate_sequence(ground_truth, predictions, *, invalid_policy="error",
                      normalized_precision=False) -> dict:
    """Score all rows, including initialization; never truncate or skip rows.

    ``error`` rejects nonfinite/nonpositive boxes. Explicit ``count_as_failure``
    retains every invalid GT/prediction row in every denominator as a failure.
    Missing/extra rows always raise. Coordinates may be zero or negative.
    Normalized precision is an opt-in supplemental diagnostic, not a primary
    published UAV/DTB metric; its reported point is <=0.20, not curve AUC.
    """
    if invalid_policy not in {"error", "count_as_failure"}:
        raise ValidationError("invalid_policy must be error or count_as_failure")
    gt = boxes_array(ground_truth, "ground_truth")
    pred = boxes_array(predictions, "predictions")
    if gt.shape != pred.shape:
        raise ValidationError(f"frame count mismatch: GT={len(gt)}, predictions={len(pred)}")
    gt_valid, pred_valid = valid_boxes(gt), valid_boxes(pred)
    valid = gt_valid & pred_valid
    if invalid_policy == "error" and not valid.all():
        raise ValidationError(f"invalid xywh rows (zero-based): GT={np.flatnonzero(~gt_valid).tolist()}, "
                              f"predictions={np.flatnonzero(~pred_valid).tolist()}")
    iou = np.full(len(gt), -1.0)
    error = np.full(len(gt), np.inf)
    norm_error = np.full(len(gt), np.inf)
    # Invalid rows are never passed into arithmetic; overflow is rejected too.
    a, b = gt[valid], pred[valid]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            a_end, b_end = a[:, :2] + a[:, 2:], b[:, :2] + b[:, 2:]
            if np.any(a_end <= a[:, :2]) or np.any(b_end <= b[:, :2]):
                raise ValidationError("box extent is not representable at this coordinate scale")
            left_top = np.maximum(a[:, :2], b[:, :2])
            right_bottom = np.minimum(a_end, b_end)
            extent = np.maximum(0.0, right_bottom - left_top)
            intersection = extent[:, 0] * extent[:, 1]
            union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - intersection
            if np.any(union <= 0):
                raise ValidationError("box area underflow or invalid union")
            iou[valid] = np.clip(intersection / union, 0.0, 1.0)
            delta = (a[:, :2] + (a[:, 2:] - 1) / 2) - (b[:, :2] + (b[:, 2:] - 1) / 2)
            error[valid] = np.hypot(delta[:, 0], delta[:, 1])
            if normalized_precision:
                norm_delta = delta / a[:, 2:]
                norm_error[valid] = np.hypot(norm_delta[:, 0], norm_delta[:, 1])
    except FloatingPointError as exc:
        raise ValidationError("box arithmetic overflow/nonfinite result") from exc
    success = (iou[:, None] > SUCCESS_THRESHOLDS).mean(axis=0)
    precision = (error[:, None] <= PRECISION_THRESHOLDS).mean(axis=0)
    normalized = ((norm_error[:, None] <= NORMALIZED_THRESHOLDS).mean(axis=0)
                  if normalized_precision else None)
    return {
        "frame_count": len(gt), "invalid_ground_truth_frames": int((~gt_valid).sum()),
        "invalid_prediction_frames": int((~pred_valid).sum()), "invalid_policy": invalid_policy,
        "initialization_frame_included": True,
        "success_auc": float(success.mean()), "precision_at_20px": float(precision[20]),
        "success_curve": success.tolist(), "precision_curve": precision.tolist(),
        "normalized_precision_at_0_20": float(normalized[20]) if normalized is not None else None,
        "normalized_precision_curve": normalized.tolist() if normalized is not None else None,
        "normalized_precision_applicability": "supplemental_diagnostic" if normalized_precision else "not_requested",
    }


def aggregate_sequences(sequences: Mapping[str, dict]) -> dict:
    """Equal sequence weighting, never pooling frames across unequal sequences."""
    if not sequences:
        raise ValidationError("cannot aggregate an empty dataset")
    rows = list(sequences.values())
    policies = {r["invalid_policy"] for r in rows}
    enabled = {r["normalized_precision_curve"] is not None for r in rows}
    if len(policies) != 1 or len(enabled) != 1:
        raise ValidationError("cannot mix metric/invalid-frame policies")
    success = np.mean([r["success_curve"] for r in rows], axis=0)
    precision = np.mean([r["precision_curve"] for r in rows], axis=0)
    norm = (np.mean([r["normalized_precision_curve"] for r in rows], axis=0)
            if enabled == {True} else None)
    return {
        "aggregation": "sequence_macro", "sequence_count": len(rows),
        "frame_count": sum(r["frame_count"] for r in rows),
        "success_auc": float(success.mean()), "precision_at_20px": float(precision[20]),
        "success_curve": success.tolist(), "precision_curve": precision.tolist(),
        "normalized_precision_at_0_20": float(norm[20]) if norm is not None else None,
        "normalized_precision_curve": norm.tolist() if norm is not None else None,
        "invalid_ground_truth_frames": sum(r["invalid_ground_truth_frames"] for r in rows),
        "invalid_prediction_frames": sum(r["invalid_prediction_frames"] for r in rows),
        "invalid_policy": next(iter(policies)),
    }
