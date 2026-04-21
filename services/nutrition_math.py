import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def generate_glucose_curve(glycemic_index: int) -> pd.DataFrame:
    """Generate a deterministic glucose curve over 120 minutes based on GI.

    Models postprandial blood glucose as a Gaussian curve peaking at ~45
    minutes.  Higher glycemic-index foods produce a taller, sharper spike.

    Args:
        glycemic_index: The glycemic index of the food (0-100 scale).

    Returns:
        A DataFrame indexed by minutes (0-120) with a single
        ``Glucose (mg/dL)`` column.
    """
    glycemic_index = max(0, min(100, int(glycemic_index)))
    minutes: np.ndarray = np.arange(0, 121)
    baseline: float = 90.0
    peak_rise: float = glycemic_index * 0.6
    spread: float = max(25 - (glycemic_index / 10), 15)
    curve: np.ndarray = baseline + peak_rise * np.exp(
        -0.5 * ((minutes - 45) / spread) ** 2
    )
    return pd.DataFrame(
        {"Minutes": minutes, "Glucose (mg/dL)": curve}
    ).set_index("Minutes")


def compute_metrics(curve_values: np.ndarray, baseline: float = 90.0) -> dict:
    """Compute peak, 2-hr AUC above baseline, and time to peak.

    Args:
        curve_values: 1-D array of glucose values sampled at 1-minute intervals.
        baseline: Fasting baseline glucose (mg/dL) to subtract before integrating.

    Returns:
        Dict with keys ``peak`` (int, mg/dL), ``time_to_peak`` (int, minutes),
        and ``auc`` (int, mg*min/dL above baseline over the full curve).
    """
    peak = float(np.max(curve_values))
    time_to_peak = int(np.argmax(curve_values))
    auc = float(np.trapezoid(np.maximum(curve_values - baseline, 0), dx=1))
    return {
        "peak": round(peak),
        "time_to_peak": time_to_peak,
        "auc": round(auc),
    }


def blend_curves(
    original: np.ndarray, swap: np.ndarray, factor: float
) -> np.ndarray:
    """Linearly interpolate between two glucose curves.

    Used to animate the what-if slider: factor=0 returns the original meal,
    factor=1 returns the full swap, anything in between is a blend.

    Args:
        original: Baseline glucose curve (1-D array).
        swap: Alternative meal's glucose curve (1-D array, same length).
        factor: Blend weight in [0, 1].

    Returns:
        Weighted-average array of the same shape.
    """
    factor = max(0.0, min(1.0, float(factor)))
    return original * (1 - factor) + swap * factor
