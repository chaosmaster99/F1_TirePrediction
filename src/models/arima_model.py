"""
Time Series Model 1: ARIMA
Autoregressive Integrated Moving Average for lap time prediction.
Models tire degradation as a time series within each stint.

Proper time-series workflow:
  1. Build average degradation curve per compound
  2. ADF test for stationarity → determine differencing order d
  3. ACF / PACF analysis on the (differenced) series → determine p, q
  4. Fit ARIMA(p,d,q) with statistically justified order
  5. Residual diagnostics (Ljung-Box)
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import logging

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_degradation_curve(fclt: np.ndarray, min_baseline_laps: int = 3) -> np.ndarray:
    """
    Convert raw FuelCorrectedLapTime array into a degradation curve.

    * Drops the first lap (outlap / pit-exit lap).
    * Baseline = mean of the next `min_baseline_laps` laps.
    * Returns deg = fclt - baseline for laps after the outlap.
    """
    if len(fclt) <= min_baseline_laps + 1:
        return np.array([])

    post_outlap = fclt[1:]  # drop outlap
    baseline = np.mean(post_outlap[:min_baseline_laps])
    deg = post_outlap - baseline
    return deg


def _smooth_curve(curve: np.ndarray, window: int = 3) -> np.ndarray:
    """Simple centred moving-average smoother (preserves length)."""
    if len(curve) < window:
        return curve
    smoothed = np.convolve(curve, np.ones(window) / window, mode="same")
    half = window // 2
    smoothed[:half] = curve[:half]
    smoothed[-half:] = curve[-half:]
    return smoothed


def _adf_test(series: np.ndarray) -> dict:
    """Run Augmented Dickey-Fuller test and return results dict.

    Uses sqrt(n) maxlag rule to ensure enough effective observations
    for reliable inference on short degradation curves.
    """
    n = len(series)
    max_lag = max(1, int(np.sqrt(n)))
    result = adfuller(series, autolag="AIC", maxlag=max_lag)
    return {
        "adf_statistic": float(result[0]),
        "p_value": float(result[1]),
        "used_lag": int(result[2]),
        "nobs": int(result[3]),
        "critical_values": {k: float(v) for k, v in result[4].items()},
        "is_stationary": result[1] < 0.05,
    }


def _kpss_test(series: np.ndarray) -> dict:
    """Run KPSS test (null = stationary)."""
    stat, p, lags, crit = kpss(series, regression="c", nlags="auto")
    return {
        "kpss_statistic": float(stat),
        "p_value": float(p),
        "used_lag": int(lags),
        "critical_values": {k: float(v) for k, v in crit.items()},
        "is_stationary": p > 0.05,  # fail to reject null → stationary
    }


def _determine_d(series: np.ndarray, max_d: int = 2) -> int:
    """Determine differencing order d via ADF + KPSS confirmation.

    Uses ADF (null: unit root) as primary test.  For short series where
    ADF lacks power, KPSS (null: stationary) serves as a secondary check:
    if KPSS fails to reject stationarity the series is accepted as
    stationary even when ADF is inconclusive.
    """
    for d in range(max_d + 1):
        diffed = series.copy()
        for _ in range(d):
            diffed = np.diff(diffed)
        if len(diffed) < 6:
            return d
        adf = _adf_test(diffed)
        if adf["is_stationary"]:
            return d
        # Secondary check: KPSS (opposite null)
        try:
            kpss_res = _kpss_test(diffed)
            if kpss_res["is_stationary"]:  # fail-to-reject H0:stationary
                return d
        except Exception:
            pass
    return max_d


def _select_order_aic(series: np.ndarray, d: int, max_p: int = 3, max_q: int = 3) -> tuple:
    """Grid-search (p, d, q) by AIC on the original series."""
    best_aic = np.inf
    best_order = (1, d, 1)
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(series, order=(p, d, q),
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                res = model.fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, d, q)
            except Exception:
                continue
    return best_order, best_aic


class ARIMADegradationModel:
    """
    ARIMA-based tire degradation model with full stationarity diagnostics.

    For each tyre compound:
      1. Builds average degradation curve from historical stints
      2. Runs ADF / KPSS stationarity tests
      3. Determines differencing order d
      4. Uses ACF/PACF + AIC grid search for (p, d, q)
      5. Fits ARIMA and runs residual diagnostics
      6. Saves diagnostic plots (ACF, PACF, stationarity, residuals)
    """

    def __init__(self, order=(1, 1, 1)):
        self.default_order = order       # fallback if auto-selection fails
        self.order = order               # will be updated per compound (last fitted)
        self.orders = {}                 # compound -> (p,d,q) actually used
        self.models = {}                 # compound -> fitted ARIMA result
        self.history = {}                # compound -> np.ndarray avg deg curve
        self.fitted_curves = {}          # compound -> ARIMA fitted values
        self.stint_curves = {}           # compound -> list of individual curves
        self.diagnostics = {}            # compound -> diagnostics dict
        self.metrics = {}
        self.name = "ARIMA"

    # ------------------------------------------------------------------
    # Stationarity diagnostics & plotting
    # ------------------------------------------------------------------
    def _run_diagnostics(self, compound: str, series: np.ndarray,
                         plots_dir: str = "plots") -> dict:
        """Run full stationarity analysis and save diagnostic plots."""
        diag = {"compound": compound, "series_length": len(series)}

        # --- ADF test on original series ---
        adf_orig = _adf_test(series)
        diag["adf_original"] = adf_orig

        # --- KPSS test on original series ---
        try:
            kpss_orig = _kpss_test(series)
        except Exception:
            kpss_orig = {"is_stationary": None, "p_value": None}
        diag["kpss_original"] = kpss_orig

        # --- Determine d ---
        d = _determine_d(series)
        diag["d_selected"] = d

        # --- Differenced series ---
        diffed = series.copy()
        for _ in range(d):
            diffed = np.diff(diffed)
        if len(diffed) >= 6:
            adf_diff = _adf_test(diffed)
            try:
                kpss_diff = _kpss_test(diffed)
            except Exception:
                kpss_diff = {"is_stationary": None, "p_value": None}
        else:
            adf_diff = {"is_stationary": True, "p_value": 0.0}
            kpss_diff = {"is_stationary": True, "p_value": 1.0}
        diag["adf_differenced"] = adf_diff
        diag["kpss_differenced"] = kpss_diff
        # Overall verdict: stationary if either test confirms
        diag["stationary_after_diff"] = (
            adf_diff.get("is_stationary", False)
            or kpss_diff.get("is_stationary", False)
        )

        # --- AIC grid search for (p, d, q) ---
        best_order, best_aic = _select_order_aic(series, d)
        diag["selected_order"] = list(best_order)
        diag["best_aic"] = float(best_aic)

        # --- Generate diagnostic plots ---
        arima_plots_dir = os.path.join(plots_dir, "arima_diagnostics")
        os.makedirs(arima_plots_dir, exist_ok=True)

        self._plot_stationarity(compound, series, diffed, d, adf_orig, adf_diff,
                                arima_plots_dir)
        self._plot_acf_pacf(compound, series, diffed, d, arima_plots_dir)

        diag["plots_dir"] = arima_plots_dir
        return diag

    def _plot_stationarity(self, compound, original, differenced, d,
                           adf_orig, adf_diff, save_dir):
        """Plot original vs differenced series with ADF results."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Original series
        axes[0].plot(original, "b-o", markersize=4, linewidth=1.5, label="Original")
        axes[0].axhline(y=np.mean(original), color="r", linestyle="--", alpha=0.6,
                        label=f"Mean = {np.mean(original):.3f}")
        stat_label = "Stationary" if adf_orig["is_stationary"] else "Non-stationary"
        axes[0].set_title(
            f"{compound} — Original Degradation Curve\n"
            f"ADF stat = {adf_orig['adf_statistic']:.4f}, "
            f"p = {adf_orig['p_value']:.4f} → {stat_label}",
            fontsize=12, fontweight="bold"
        )
        axes[0].set_xlabel("Lap (within stint)")
        axes[0].set_ylabel("Degradation (s)")
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Differenced series
        axes[1].plot(differenced, "g-s", markersize=4, linewidth=1.5,
                     label=f"Differenced (d={d})")
        axes[1].axhline(y=0, color="r", linestyle="--", alpha=0.6)
        if isinstance(adf_diff.get("adf_statistic"), (int, float)):
            stat_label2 = "Stationary" if adf_diff["is_stationary"] else "Non-stationary"
            axes[1].set_title(
                f"{compound} — After Differencing (d={d})\n"
                f"ADF stat = {adf_diff['adf_statistic']:.4f}, "
                f"p = {adf_diff['p_value']:.4f} → {stat_label2}",
                fontsize=12, fontweight="bold"
            )
        else:
            axes[1].set_title(f"{compound} — After Differencing (d={d})", fontsize=12)
        axes[1].set_xlabel("Lap")
        axes[1].set_ylabel("Differenced Value")
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(save_dir, f"stationarity_{compound.lower()}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Stationarity plot saved: {path}")

    def _plot_acf_pacf(self, compound, original, differenced, d, save_dir):
        """Plot ACF and PACF for both original and differenced series."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        max_lags_orig = min(len(original) // 2 - 1, 15)
        max_lags_diff = min(len(differenced) // 2 - 1, 15)
        if max_lags_orig < 1:
            max_lags_orig = 1
        if max_lags_diff < 1:
            max_lags_diff = 1

        # ACF — Original
        plot_acf(original, ax=axes[0, 0], lags=max_lags_orig, alpha=0.05)
        axes[0, 0].set_title(f"{compound} — ACF (Original)", fontweight="bold")

        # PACF — Original
        plot_pacf(original, ax=axes[0, 1], lags=max_lags_orig, alpha=0.05,
                  method="ywm")
        axes[0, 1].set_title(f"{compound} — PACF (Original)", fontweight="bold")

        # ACF — Differenced
        plot_acf(differenced, ax=axes[1, 0], lags=max_lags_diff, alpha=0.05)
        axes[1, 0].set_title(f"{compound} — ACF (d={d})", fontweight="bold")

        # PACF — Differenced
        plot_pacf(differenced, ax=axes[1, 1], lags=max_lags_diff, alpha=0.05,
                  method="ywm")
        axes[1, 1].set_title(f"{compound} — PACF (d={d})", fontweight="bold")

        for ax in axes.flat:
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"ACF / PACF Analysis — {compound} Compound\n"
            f"(Top: original series, Bottom: after d={d} differencing)",
            fontsize=13, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        path = os.path.join(save_dir, f"acf_pacf_{compound.lower()}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  ACF/PACF plot saved: {path}")

    def _plot_residuals(self, compound, residuals, save_dir):
        """Plot residual diagnostics for the fitted ARIMA."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Residual time plot
        axes[0, 0].plot(residuals, "b-o", markersize=3, linewidth=1)
        axes[0, 0].axhline(y=0, color="r", linestyle="--")
        axes[0, 0].set_title(f"{compound} — Residuals Over Time", fontweight="bold")
        axes[0, 0].set_xlabel("Lap")
        axes[0, 0].set_ylabel("Residual")
        axes[0, 0].grid(True, alpha=0.3)

        # Residual histogram
        axes[0, 1].hist(residuals, bins=15, color="#3498db", edgecolor="black",
                        density=True, alpha=0.7)
        axes[0, 1].set_title(f"{compound} — Residual Distribution", fontweight="bold")
        axes[0, 1].set_xlabel("Residual")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].grid(True, alpha=0.3)

        # ACF of residuals
        max_lags = min(len(residuals) // 2 - 1, 12)
        if max_lags < 1:
            max_lags = 1
        plot_acf(residuals, ax=axes[1, 0], lags=max_lags, alpha=0.05)
        axes[1, 0].set_title(f"{compound} — ACF of Residuals", fontweight="bold")

        # QQ-ish: residuals vs fitted order
        axes[1, 1].scatter(range(len(residuals)), np.sort(residuals),
                           c="#e74c3c", s=20, alpha=0.7)
        axes[1, 1].axhline(y=0, color="gray", linestyle="--")
        axes[1, 1].set_title(f"{compound} — Sorted Residuals", fontweight="bold")
        axes[1, 1].set_xlabel("Order")
        axes[1, 1].set_ylabel("Residual")
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f"Residual Diagnostics — {compound} ARIMA",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        path = os.path.join(save_dir, f"residuals_{compound.lower()}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Residual plot saved: {path}")

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, ts_data: dict, processed_df: pd.DataFrame = None):
        """
        Fit one ARIMA model per tyre compound with full diagnostics.

        Steps per compound:
          1. Collect & clean degradation curves
          2. Build average curve
          3. Run ADF/KPSS stationarity tests
          4. ACF/PACF analysis → determine (p, d, q)
          5. Fit ARIMA with auto-selected order
          6. Residual diagnostics
        """
        logger.info(f"Fitting {self.name} on {len(ts_data)} stints …")

        # 1. Collect clean degradation curves grouped by compound
        curves_by_compound: dict[str, list[np.ndarray]] = {}

        for stint_key, stint_df in ts_data.items():
            compound = stint_df["Compound"].iloc[0]
            fclt = stint_df["FuelCorrectedLapTime"].values

            deg = _clean_degradation_curve(fclt)
            if len(deg) < 5:
                continue

            curves_by_compound.setdefault(compound, []).append(deg)

        # 2. Build average curve per compound, run diagnostics, and fit ARIMA
        for compound, curves in curves_by_compound.items():
            self.stint_curves[compound] = curves

            # Align to common length
            median_len = int(np.median([len(c) for c in curves]))
            target_len = max(10, min(40, median_len))
            truncated = [c[:target_len] for c in curves if len(c) >= target_len]

            if len(truncated) < 5:
                max_len = min(40, max(len(c) for c in curves))
                padded = []
                for c in curves:
                    if len(c) >= max_len:
                        padded.append(c[:max_len])
                    else:
                        pad = np.full(max_len - len(c), c[-1])
                        padded.append(np.concatenate([c, pad]))
                avg_curve = np.mean(padded, axis=0)
            else:
                avg_curve = np.mean(truncated, axis=0)

            # Smooth to remove lap-to-lap noise
            avg_curve = _smooth_curve(avg_curve, window=3)

            # Ensure monotonically non-decreasing
            for i in range(1, len(avg_curve)):
                if avg_curve[i] < avg_curve[i - 1]:
                    avg_curve[i] = avg_curve[i - 1]

            self.history[compound] = avg_curve

            # --- 3-4. Stationarity diagnostics + order selection ---
            logger.info(f"  [{compound}] Running stationarity diagnostics …")
            diag = self._run_diagnostics(compound, avg_curve)
            self.diagnostics[compound] = diag

            selected_order = tuple(diag["selected_order"])
            self.orders[compound] = selected_order
            self.order = selected_order  # keep last for backward compat

            logger.info(
                f"  [{compound}] ADF p={diag['adf_original']['p_value']:.4f}, "
                f"d={diag['d_selected']}, "
                f"Selected order: {selected_order}, AIC={diag['best_aic']:.2f}"
            )

            # --- 5. Fit ARIMA ---
            try:
                model = ARIMA(
                    avg_curve,
                    order=selected_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit()
                self.models[compound] = fitted

                fv = np.asarray(fitted.fittedvalues, dtype=float)
                if len(fv) > 0 and np.isnan(fv[0]):
                    fv[0] = avg_curve[0]
                self.fitted_curves[compound] = fv

                # --- 6. Residual diagnostics ---
                residuals = fitted.resid
                arima_plots_dir = os.path.join("plots", "arima_diagnostics")
                os.makedirs(arima_plots_dir, exist_ok=True)
                self._plot_residuals(compound, residuals, arima_plots_dir)

                # Ljung-Box test on residuals
                try:
                    lb = acorr_ljungbox(residuals, lags=[5, 10], return_df=True)
                    diag["ljung_box"] = {
                        str(int(row.name)): {
                            "statistic": float(row["lb_stat"]),
                            "p_value": float(row["lb_pvalue"]),
                        }
                        for _, row in lb.iterrows()
                    }
                except Exception:
                    pass

                logger.info(
                    f"  [{compound}] ARIMA{selected_order} fitted — "
                    f"{len(truncated) if truncated else len(curves)} stints, "
                    f"curve len={len(avg_curve)}, AIC={fitted.aic:.2f}"
                )
            except Exception as e:
                logger.warning(f"  Failed to fit {self.name} for {compound}: {e}")

        # Save diagnostics summary JSON
        self._save_diagnostics_json()

        logger.info(
            f"{self.name} fitting complete — models for: {list(self.models.keys())}"
        )

    def _save_diagnostics_json(self):
        """Save diagnostics summary to plots/arima_diagnostics/diagnostics.json."""
        out_dir = os.path.join("plots", "arima_diagnostics")
        os.makedirs(out_dir, exist_ok=True)

        def _make_serializable(obj):
            """Convert numpy types to native Python for JSON."""
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_make_serializable(v) for v in obj]
            return obj

        summary = {}
        for compound, diag in self.diagnostics.items():
            entry = {
                "series_length": diag.get("series_length"),
                "adf_original": diag.get("adf_original"),
                "kpss_original": diag.get("kpss_original"),
                "d_selected": diag.get("d_selected"),
                "adf_differenced": diag.get("adf_differenced"),
                "kpss_differenced": diag.get("kpss_differenced"),
                "stationary_after_diff": diag.get("stationary_after_diff"),
                "selected_order": diag.get("selected_order"),
                "best_aic": diag.get("best_aic"),
                "ljung_box": diag.get("ljung_box", {}),
            }
            summary[compound] = _make_serializable(entry)

        path = os.path.join(out_dir, "diagnostics.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Diagnostics JSON saved to {path}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_degradation(self, compound: str, current_tyre_life: int,
                            forecast_laps: int = 15) -> np.ndarray:
        """
        Predict degradation for the next *forecast_laps* laps.

        Strategy:
        - For laps within the stored average curve → read directly.
        - For laps beyond it → use the full-history ARIMA to forecast.

        Returns an array of length `forecast_laps` giving the expected
        time loss (seconds) relative to fresh-tyre pace.
        """
        if compound not in self.models:
            logger.warning(f"No {self.name} model for {compound} — using fallback")
            return self._fallback_predict(compound, current_tyre_life, forecast_laps)

        history = self.history[compound]
        fitted_model = self.models[compound]

        try:
            result = []
            need_forecast = 0

            for i in range(forecast_laps):
                future_life = current_tyre_life + i + 1
                hist_idx = future_life - 2
                if hist_idx < 0:
                    result.append(0.0)
                elif hist_idx < len(history):
                    result.append(history[hist_idx])
                else:
                    need_forecast = forecast_laps - i
                    break

            if need_forecast > 0:
                first_beyond_life = current_tyre_life + (forecast_laps - need_forecast) + 1
                first_beyond_idx = first_beyond_life - 2
                steps_past_start = first_beyond_idx - len(history)
                total_steps = steps_past_start + need_forecast

                forecast = fitted_model.forecast(steps=total_steps)
                forecast = np.maximum(forecast, 0.0)
                relevant = forecast[steps_past_start:]

                last_val = history[-1] if len(history) > 0 else 0.0
                for val in relevant:
                    clamped = max(val, last_val)
                    result.append(clamped)
                    last_val = clamped

            arr = np.array(result[:forecast_laps])
            arr = np.maximum(arr, 0.0)
            arr = np.minimum(arr, 5.0)
            return arr

        except Exception as e:
            logger.warning(f"ARIMA forecast failed for {compound}: {e}")
            return self._fallback_predict(compound, current_tyre_life, forecast_laps)

    def _fallback_predict(self, compound: str, current_tyre_life: int,
                          forecast_laps: int) -> np.ndarray:
        """Simple quadratic degradation fallback."""
        rates = {"SOFT": 0.08, "MEDIUM": 0.05, "HARD": 0.03}
        rate = rates.get(compound, 0.05)
        b = rate * 0.6
        a = rate * 0.03
        return np.array([
            a * (current_tyre_life + i + 1) ** 2 + b * (current_tyre_life + i + 1)
            for i in range(forecast_laps)
        ])

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, ts_data: dict) -> dict:
        """
        Evaluate model against individual stints.

        For each stint the model's average-curve prediction is compared
        against the stint's actual (clean) degradation at each lap.
        """
        all_actual = []
        all_predicted = []

        for stint_key, stint_df in ts_data.items():
            compound = stint_df["Compound"].iloc[0]
            if compound not in self.models:
                continue

            fclt = stint_df["FuelCorrectedLapTime"].values
            deg = _clean_degradation_curve(fclt)
            if len(deg) < 5:
                continue

            curve = self.fitted_curves.get(compound, self.history.get(compound))
            if curve is None:
                continue
            n = min(len(deg), len(curve))
            if n < 3:
                continue

            all_actual.extend(deg[:n].tolist())
            all_predicted.extend(curve[:n].tolist())

        if not all_actual:
            logger.warning(f"{self.name} evaluation: no valid predictions produced")
            return {"rmse": float("inf"), "mae": float("inf"), "r2": -1.0}

        actual_arr = np.array(all_actual)
        pred_arr = np.array(all_predicted)

        self.metrics = {
            "rmse": float(np.sqrt(mean_squared_error(actual_arr, pred_arr))),
            "mae": float(mean_absolute_error(actual_arr, pred_arr)),
            "r2": float(r2_score(actual_arr, pred_arr)) if len(actual_arr) > 1 else 0.0,
        }

        logger.info(
            f"{self.name} Evaluation — RMSE: {self.metrics['rmse']:.4f}, "
            f"MAE: {self.metrics['mae']:.4f}, R²: {self.metrics['r2']:.4f}"
        )
        return self.metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save model artefacts to disk."""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.history, os.path.join(path, "arima_history.pkl"))
        joblib.dump(self.models, os.path.join(path, "arima_models.pkl"))
        joblib.dump(self.order, os.path.join(path, "arima_order.pkl"))
        joblib.dump(self.metrics, os.path.join(path, "arima_metrics.pkl"))
        joblib.dump(self.fitted_curves, os.path.join(path, "arima_fitted.pkl"))
        joblib.dump(self.diagnostics, os.path.join(path, "arima_diagnostics.pkl"))
        joblib.dump(self.orders, os.path.join(path, "arima_orders.pkl"))
        logger.info(f"ARIMA saved to {path}")

    def load(self, path: str):
        """Load model artefacts from disk."""
        self.history = joblib.load(os.path.join(path, "arima_history.pkl"))
        self.models = joblib.load(os.path.join(path, "arima_models.pkl"))
        self.order = joblib.load(os.path.join(path, "arima_order.pkl"))
        metrics_path = os.path.join(path, "arima_metrics.pkl")
        if os.path.exists(metrics_path):
            self.metrics = joblib.load(metrics_path)
        fitted_path = os.path.join(path, "arima_fitted.pkl")
        if os.path.exists(fitted_path):
            self.fitted_curves = joblib.load(fitted_path)
        diag_path = os.path.join(path, "arima_diagnostics.pkl")
        if os.path.exists(diag_path):
            self.diagnostics = joblib.load(diag_path)
        orders_path = os.path.join(path, "arima_orders.pkl")
        if os.path.exists(orders_path):
            self.orders = joblib.load(orders_path)
        logger.info(f"ARIMA loaded from {path} — compounds: {list(self.models.keys())}")

    # ------------------------------------------------------------------
    # Strategy helper
    # ------------------------------------------------------------------
    def get_pit_recommendation(self, compound: str, current_tyre_life: int,
                                threshold: float = 1.5, max_forecast: int = 30) -> int:
        """
        Recommend laps until pit based on degradation threshold.

        Returns:
            Number of laps until recommended pit stop (-1 if no pit needed).
        """
        forecast = self.predict_degradation(compound, current_tyre_life, max_forecast)

        for i, deg in enumerate(forecast):
            if deg > threshold:
                return i + 1

        return -1  # No pit needed within forecast window
