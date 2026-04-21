import numpy as np
import pytest

from services.nutrition_math import (
    generate_glucose_curve,
    compute_metrics,
    blend_curves,
)


# ---------------------------------------------------------------------------
# generate_glucose_curve
# ---------------------------------------------------------------------------
class TestGenerateGlucoseCurve:
    def test_returns_121_rows(self):
        df = generate_glucose_curve(60)
        assert len(df) == 121

    def test_baseline_at_zero_gi(self):
        df = generate_glucose_curve(0)
        values = df["Glucose (mg/dL)"].values
        # With GI 0, peak_rise is 0 → flat line at baseline 90
        np.testing.assert_allclose(values, 90.0)

    def test_higher_gi_means_higher_peak(self):
        low = generate_glucose_curve(30)["Glucose (mg/dL)"].max()
        high = generate_glucose_curve(80)["Glucose (mg/dL)"].max()
        assert high > low

    def test_peak_near_45_minutes(self):
        df = generate_glucose_curve(70)
        peak_minute = df["Glucose (mg/dL)"].idxmax()
        assert 40 <= peak_minute <= 50

    def test_clamps_gi_above_100(self):
        clamped = generate_glucose_curve(150)
        normal = generate_glucose_curve(100)
        np.testing.assert_array_equal(
            clamped["Glucose (mg/dL)"].values,
            normal["Glucose (mg/dL)"].values,
        )

    def test_clamps_gi_below_0(self):
        clamped = generate_glucose_curve(-10)
        normal = generate_glucose_curve(0)
        np.testing.assert_array_equal(
            clamped["Glucose (mg/dL)"].values,
            normal["Glucose (mg/dL)"].values,
        )


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------
class TestComputeMetrics:
    def test_flat_baseline_curve(self):
        flat = np.full(121, 90.0)
        m = compute_metrics(flat, baseline=90.0)
        assert m["peak"] == 90
        assert m["auc"] == 0
        assert m["time_to_peak"] == 0  # argmax of constant returns 0

    def test_peak_value(self):
        curve = np.full(121, 90.0)
        curve[45] = 150.0
        m = compute_metrics(curve, baseline=90.0)
        assert m["peak"] == 150

    def test_time_to_peak(self):
        curve = np.full(121, 90.0)
        curve[60] = 140.0
        m = compute_metrics(curve, baseline=90.0)
        assert m["time_to_peak"] == 60

    def test_auc_positive_only(self):
        # Values below baseline should not contribute negative AUC
        curve = np.full(121, 80.0)  # all below baseline
        m = compute_metrics(curve, baseline=90.0)
        assert m["auc"] == 0


# ---------------------------------------------------------------------------
# blend_curves
# ---------------------------------------------------------------------------
class TestBlendCurves:
    def setup_method(self):
        self.a = np.full(121, 100.0)
        self.b = np.full(121, 80.0)

    def test_factor_zero_returns_original(self):
        result = blend_curves(self.a, self.b, 0.0)
        np.testing.assert_array_equal(result, self.a)

    def test_factor_one_returns_swap(self):
        result = blend_curves(self.a, self.b, 1.0)
        np.testing.assert_array_equal(result, self.b)

    def test_factor_half_returns_midpoint(self):
        result = blend_curves(self.a, self.b, 0.5)
        np.testing.assert_allclose(result, 90.0)

    def test_clamps_factor_above_one(self):
        result = blend_curves(self.a, self.b, 1.5)
        np.testing.assert_array_equal(result, self.b)

    def test_clamps_factor_below_zero(self):
        result = blend_curves(self.a, self.b, -0.5)
        np.testing.assert_array_equal(result, self.a)

    def test_output_shape_matches_input(self):
        result = blend_curves(self.a, self.b, 0.3)
        assert result.shape == self.a.shape
