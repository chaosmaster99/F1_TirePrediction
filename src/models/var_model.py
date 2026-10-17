"""
Time Series Model 2: VAR (Vector Autoregression)
Multivariate time series model for tire degradation forecasting.

Models the degradation curves of SOFT, MEDIUM, and HARD compounds
as a system of interdependent equations — capturing cross-compound
dynamics (e.g. if SOFT degrades faster, MEDIUM likely follows a
similar structural pattern).

Degradation is defined as the increase in fuel-corrected lap time
relative to a *settled-pace baseline* (mean of stint laps 2-4),
identical to the ARIMA model definition.
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import logging

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers (identical to arima_model.py for consistency)
# ---------------------------------------------------------------------------

def _clean_degradation_curve(fclt: np.ndarray, min_baseline_laps: int = 3) -> np.ndarray:
    if len(fclt) <= min_baseline_laps + 1:
        return np.array([])
    post_outlap = fclt[1:]
    baseline = np.mean(post_outlap[:min_baseline_laps])
    deg = post_outlap - baseline
    return deg


def _smooth_curve(curve: np.ndarray, window: int = 3) -> np.ndarray:
    if len(curve) < window:
        return curve
    smoothed = np.convolve(curve, np.ones(window) / window, mode="same")
    half = window // 2
    smoothed[:half] = curve[:half]
    smoothed[-half:] = curve[-half:]
    return smoothed


class VARDegradationModel:
    """
    Vector Autoregression model for tire degradation.

    Treats the three tyre compounds (SOFT, MEDIUM, HARD) as a
    multivariate system of shape (T, 3).  A single VAR(p) model
    is fitted on the aligned, averaged degradation curves, capturing
    cross-compound lead-lag relationships.
    """

    COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

    def __init__(self, maxlags: int = 5):
        self.maxlags = maxlags
        self.var_model = None         # fitted VARResults
        self.selected_lag = None      # p chosen by AIC
        self.history = {}             # compound -> avg degradation curve
        self.aligned_matrix = None    # (T, 3) ndarray used for fitting
        self.aligned_length = None    # T
        self.fitted_curves = {}       # compound -> fitted values array
        self.stint_curves = {}        # compound -> list of individual curves
        self.diagnostics = {}
        self.metrics = {}
        self.name = "VAR"

    # ------------------------------------------------------------------
    # Curve collection (shared logic with ARIMA/ETS)
    # ------------------------------------------------------------------
    def _collect_curves(self, ts_data: dict) -> dict:
        """Collect cleaned degradation curves grouped by compound."""
        curves_by_compound: dict[str, list[np.ndarray]] = {}
        for stint_key, stint_df in ts_data.items():
            compound = stint_df["Compound"].iloc[0]
            fclt = stint_df["FuelCorrectedLapTime"].values
            deg = _clean_degradation_curve(fclt)
            if len(deg) < 5:
                continue
            curves_by_compound.setdefault(compound, []).append(deg)
        return curves_by_compound

    def _build_avg_curve(self, curves: list) -> np.ndarray:
        """Build smoothed, monotonically non-decreasing average curve."""
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

        avg_curve = _smooth_curve(avg_curve, window=3)
        for i in range(1, len(avg_curve)):
            if avg_curve[i] < avg_curve[i - 1]:
                avg_curve[i] = avg_curve[i - 1]
        return avg_curve

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, ts_data: dict, processed_df: pd.DataFrame = None):
        """
        Fit a VAR(p) model on aligned, averaged degradation curves.

        1. Collect & clean degradation curves per compound.
        2. Build average degradation curve for each compound.
        3. Align to common length → (T, 3) matrix.
        4. Select lag order p via AIC (up to maxlags).
        5. Fit VAR(p).
        6. Generate diagnostic plots.
        """
        logger.info(f"Fitting {self.name} on {len(ts_data)} stints …")

        curves_by_compound = self._collect_curves(ts_data)

        # Build average curve per compound
        for compound in self.COMPOUNDS:
            if compound not in curves_by_compound:
                logger.warning(f"  No curves for {compound} — skipping")
                continue
            curves = curves_by_compound[compound]
            self.stint_curves[compound] = curves
            avg = self._build_avg_curve(curves)
            self.history[compound] = avg
            logger.info(f"  {compound}: {len(curves)} stints, avg curve len={len(avg)}")

        # Need all 3 compounds for a proper VAR
        available = [c for c in self.COMPOUNDS if c in self.history]
        if len(available) < 2:
            logger.warning("VAR requires at least 2 compounds — falling back to independent fits")
            return

        # Align to shortest common length
        common_len = min(len(self.history[c]) for c in available)
        common_len = max(common_len, 8)  # need enough observations for VAR
        matrix = np.column_stack([self.history[c][:common_len] for c in available])
        self.aligned_matrix = matrix
        self.aligned_length = common_len

        # Create DataFrame for statsmodels VAR
        df_var = pd.DataFrame(matrix, columns=available)

        # Select lag order via AIC
        try:
            var_model = VAR(df_var)
            k = len(available)  # number of endogenous variables
            # VAR needs T > k*p + 1 minimum; be conservative
            max_possible_lag = max(1, (common_len - 1) // (k + 1) - 1)
            test_lags = min(self.maxlags, max_possible_lag)

            if test_lags >= 2:
                try:
                    selection = var_model.select_order(maxlags=test_lags)
                    self.selected_lag = selection.aic
                    logger.info(f"  VAR lag selection — AIC: p={selection.aic}, "
                                f"BIC: p={selection.bic}, HQIC: p={selection.hqic}")
                except Exception as lag_err:
                    logger.warning(f"  VAR select_order failed ({lag_err}), using p=1")
                    self.selected_lag = 1
            else:
                self.selected_lag = 1

            p = max(1, self.selected_lag)
            fitted = var_model.fit(p)
            self.var_model = fitted
            self.diagnostics["lag_order"] = p
            self.diagnostics["aic"] = float(fitted.aic)
            self.diagnostics["bic"] = float(fitted.bic)
            self.diagnostics["compounds_used"] = available

            # Extract fitted values
            fv = fitted.fittedvalues
            for i, compound in enumerate(available):
                vals = np.zeros(common_len)
                # fittedvalues start from index p (lag)
                if isinstance(fv, pd.DataFrame):
                    fit_arr = fv[compound].values
                else:
                    fit_arr = fv[:, i]
                vals[:p] = self.history[compound][:p]
                vals[p:p + len(fit_arr)] = fit_arr
                self.fitted_curves[compound] = vals

            logger.info(f"  VAR({p}) fitted — AIC={fitted.aic:.2f}, "
                        f"BIC={fitted.bic:.2f}, compounds={available}")

            # Run diagnostics & plots
            self._run_diagnostics(df_var, fitted, available)

        except Exception as e:
            logger.warning(f"VAR fitting failed: {e} — falling back to compound-independent curves")
            for compound in available:
                self.fitted_curves[compound] = self.history[compound][:common_len]

        logger.info(f"{self.name} fitting complete — compounds: {list(self.history.keys())}")

    # ------------------------------------------------------------------
    # Diagnostics & Plotting
    # ------------------------------------------------------------------
    def _run_diagnostics(self, df_var: pd.DataFrame, fitted, compounds: list):
        """Run VAR diagnostics and generate plots."""
        var_plots_dir = os.path.join("plots", "var_diagnostics")
        os.makedirs(var_plots_dir, exist_ok=True)

        # 1. Cross-correlation matrix plot
        self._plot_cross_correlation(df_var, compounds, var_plots_dir)

        # 2. Impulse Response Functions
        self._plot_impulse_response(fitted, compounds, var_plots_dir)

        # 3. Fitted vs Actual overlay
        self._plot_fitted_vs_actual(compounds, var_plots_dir)

        # 4. Residual diagnostics
        self._plot_residuals(fitted, compounds, var_plots_dir)

        # 5. Lag order selection criteria
        self._plot_lag_selection(df_var, var_plots_dir)

        # 6. Granger causality summary
        self._granger_causality(fitted, compounds, var_plots_dir)

        # Save diagnostics JSON
        self._save_diagnostics_json(var_plots_dir)

    def _plot_cross_correlation(self, df_var, compounds, save_dir):
        """Plot cross-correlation heatmap between compounds."""
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = df_var.corr()
        im = ax.imshow(corr.values, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(compounds)))
        ax.set_yticks(range(len(compounds)))
        ax.set_xticklabels(compounds, fontsize=12)
        ax.set_yticklabels(compounds, fontsize=12)
        for i in range(len(compounds)):
            for j in range(len(compounds)):
                ax.text(j, i, f"{corr.values[i, j]:.3f}", ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="white" if abs(corr.values[i, j]) > 0.5 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Cross-Compound Degradation Correlation", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(save_dir, "cross_correlation.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Cross-correlation plot saved: {path}")

    def _plot_impulse_response(self, fitted, compounds, save_dir):
        """Plot Impulse Response Functions (IRF)."""
        try:
            irf = fitted.irf(periods=10)
            n_compounds = len(compounds)
            fig, axes = plt.subplots(n_compounds, n_compounds,
                                     figsize=(5 * n_compounds, 4 * n_compounds))
            if n_compounds == 1:
                axes = np.array([[axes]])

            for i, resp in enumerate(compounds):
                for j, impulse in enumerate(compounds):
                    ax = axes[i][j]
                    response_data = irf.irfs[:, i, j]
                    ax.plot(response_data, "b-o", markersize=4, linewidth=2)
                    ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
                    ax.fill_between(range(len(response_data)), response_data,
                                    alpha=0.2, color="blue")
                    ax.set_title(f"Impulse: {impulse} → Response: {resp}",
                                 fontsize=10, fontweight="bold")
                    ax.set_xlabel("Periods")
                    ax.set_ylabel("Response")
                    ax.grid(True, alpha=0.3)

            plt.suptitle("Impulse Response Functions\n(How a shock to one compound affects others)",
                         fontsize=14, fontweight="bold", y=1.02)
            plt.tight_layout()
            path = os.path.join(save_dir, "impulse_response.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"  IRF plot saved: {path}")
        except Exception as e:
            logger.warning(f"  IRF plot failed: {e}")

    def _plot_fitted_vs_actual(self, compounds, save_dir):
        """Plot fitted vs actual degradation for each compound."""
        fig, axes = plt.subplots(1, len(compounds), figsize=(6 * len(compounds), 5))
        if len(compounds) == 1:
            axes = [axes]

        colors = {"SOFT": "#FF0000", "MEDIUM": "#FFD700", "HARD": "#AAAAAA"}
        for idx, compound in enumerate(compounds):
            ax = axes[idx]
            actual = self.history[compound][:self.aligned_length]
            fitted = self.fitted_curves.get(compound, actual)[:self.aligned_length]

            laps = np.arange(1, len(actual) + 1)
            ax.plot(laps, actual, "o-", color=colors.get(compound, "#333"),
                    label="Actual (avg)", linewidth=2, markersize=5)
            ax.plot(laps, fitted, "s--", color="#3498DB",
                    label="VAR Fitted", linewidth=2, markersize=4, alpha=0.8)
            ax.fill_between(laps, actual, fitted, alpha=0.15, color="#3498DB")
            ax.set_title(f"{compound} — Fitted vs Actual", fontsize=13, fontweight="bold")
            ax.set_xlabel("Stint Lap")
            ax.set_ylabel("Degradation (s)")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.suptitle("VAR Model: Fitted vs Actual Degradation Curves",
                      fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        path = os.path.join(save_dir, "fitted_vs_actual.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Fitted vs Actual plot saved: {path}")

    def _plot_residuals(self, fitted, compounds, save_dir):
        """Plot residual diagnostics for each compound."""
        resid = fitted.resid
        n = len(compounds)
        fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        for idx, compound in enumerate(compounds):
            if isinstance(resid, pd.DataFrame):
                r = resid[compound].values
            else:
                r = resid[:, idx]

            # Time plot
            axes[idx, 0].plot(r, "b-o", markersize=3, linewidth=1)
            axes[idx, 0].axhline(y=0, color="r", linestyle="--")
            axes[idx, 0].set_title(f"{compound} — Residuals Over Time", fontweight="bold")
            axes[idx, 0].set_xlabel("Lap")
            axes[idx, 0].set_ylabel("Residual")
            axes[idx, 0].grid(True, alpha=0.3)

            # Histogram
            axes[idx, 1].hist(r, bins=15, color="#3498db", edgecolor="black",
                              density=True, alpha=0.7)
            axes[idx, 1].set_title(f"{compound} — Residual Distribution", fontweight="bold")
            axes[idx, 1].set_xlabel("Residual")
            axes[idx, 1].set_ylabel("Density")
            axes[idx, 1].grid(True, alpha=0.3)

            # ACF of residuals
            from statsmodels.graphics.tsaplots import plot_acf
            max_lags = min(len(r) // 2 - 1, 10)
            if max_lags >= 1:
                plot_acf(r, ax=axes[idx, 2], lags=max_lags, alpha=0.05)
            axes[idx, 2].set_title(f"{compound} — ACF of Residuals", fontweight="bold")

        plt.suptitle("VAR Residual Diagnostics", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        path = os.path.join(save_dir, "residuals.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Residual diagnostics plot saved: {path}")

    def _plot_lag_selection(self, df_var, save_dir):
        """Plot information criteria vs lag order."""
        try:
            var_model = VAR(df_var)
            common_len = len(df_var)
            k = df_var.shape[1]
            max_possible = max(1, (common_len - 1) // (k + 1) - 1)
            test_lags = min(self.maxlags, max_possible)
            if test_lags < 1:
                return

            aics, bics, hqics = [], [], []
            lags_range = range(1, test_lags + 1)
            for p in lags_range:
                try:
                    res = var_model.fit(p)
                    aics.append(res.aic)
                    bics.append(res.bic)
                    hqics.append(res.hqic)
                except Exception:
                    aics.append(np.nan)
                    bics.append(np.nan)
                    hqics.append(np.nan)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(list(lags_range), aics, "o-", label="AIC", linewidth=2, markersize=6)
            ax.plot(list(lags_range), bics, "s-", label="BIC", linewidth=2, markersize=6)
            ax.plot(list(lags_range), hqics, "^-", label="HQIC", linewidth=2, markersize=6)

            best_p = list(lags_range)[np.nanargmin(aics)]
            ax.axvline(x=best_p, color="red", linestyle="--", alpha=0.7,
                       label=f"Best AIC: p={best_p}")

            ax.set_xlabel("Lag Order (p)", fontsize=12)
            ax.set_ylabel("Information Criterion", fontsize=12)
            ax.set_title("VAR Lag Order Selection", fontsize=14, fontweight="bold")
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(list(lags_range))

            plt.tight_layout()
            path = os.path.join(save_dir, "lag_selection.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"  Lag selection plot saved: {path}")
        except Exception as e:
            logger.warning(f"  Lag selection plot failed: {e}")

    def _granger_causality(self, fitted, compounds, save_dir):
        """Test and log Granger causality between compounds."""
        try:
            gc_results = {}
            for caused in compounds:
                for causing in compounds:
                    if caused == causing:
                        continue
                    try:
                        result = fitted.test_causality(caused, [causing], kind="f")
                        gc_results[f"{causing}→{caused}"] = {
                            "f_statistic": float(result.test_statistic),
                            "p_value": float(result.pvalue),
                            "significant": result.pvalue < 0.05,
                        }
                    except Exception:
                        pass

            self.diagnostics["granger_causality"] = gc_results

            # Plot as heatmap
            if gc_results:
                n = len(compounds)
                pval_matrix = np.ones((n, n))
                for i, caused in enumerate(compounds):
                    for j, causing in enumerate(compounds):
                        key = f"{causing}→{caused}"
                        if key in gc_results:
                            pval_matrix[i, j] = gc_results[key]["p_value"]

                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(pval_matrix, cmap="RdYlGn", vmin=0, vmax=0.2, aspect="auto")
                ax.set_xticks(range(n))
                ax.set_yticks(range(n))
                ax.set_xticklabels([f"{c}\n(causing)" for c in compounds], fontsize=11)
                ax.set_yticklabels([f"{c}\n(caused)" for c in compounds], fontsize=11)
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            ax.text(j, i, "—", ha="center", va="center", fontsize=12)
                        else:
                            p = pval_matrix[i, j]
                            sig = "✓" if p < 0.05 else "✗"
                            ax.text(j, i, f"p={p:.3f}\n{sig}", ha="center", va="center",
                                    fontsize=10, fontweight="bold",
                                    color="white" if p < 0.1 else "black")
                plt.colorbar(im, ax=ax, shrink=0.8, label="p-value")
                ax.set_title("Granger Causality Test (p < 0.05 = significant)",
                             fontsize=13, fontweight="bold")
                plt.tight_layout()
                path = os.path.join(save_dir, "granger_causality.png")
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info(f"  Granger causality plot saved: {path}")

        except Exception as e:
            logger.warning(f"  Granger causality test failed: {e}")

    def _save_diagnostics_json(self, save_dir):
        """Save diagnostics to JSON."""
        def _serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_serializable(v) for v in obj]
            return obj

        path = os.path.join(save_dir, "diagnostics.json")
        with open(path, "w") as f:
            json.dump(_serializable(self.diagnostics), f, indent=2)
        logger.info(f"  VAR diagnostics JSON saved: {path}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_degradation(self, compound: str, current_tyre_life: int,
                            forecast_laps: int = 15) -> np.ndarray:
        """
        Predict degradation for the next *forecast_laps* laps.

        Strategy (matches ARIMA):
        - For laps within the stored average curve → read directly.
        - For laps beyond → use VAR to forecast the system.
        - Cap at 5.0 s to prevent runaway extrapolation.
        """
        if compound not in self.history:
            logger.warning(f"No {self.name} data for {compound} — using fallback")
            return self._fallback_predict(compound, current_tyre_life, forecast_laps)

        history = self.history[compound]
        available = [c for c in self.COMPOUNDS if c in self.history]
        comp_idx = available.index(compound) if compound in available else 0

        try:
            result = []
            need_forecast = 0

            for i in range(forecast_laps):
                future_life = current_tyre_life + i + 1
                hist_idx = future_life - 2
                if hist_idx < 0:
                    result.append(0.0)
                elif hist_idx < len(history):
                    result.append(max(history[hist_idx], 0.0))
                else:
                    need_forecast = forecast_laps - i
                    break

            if need_forecast > 0 and self.var_model is not None:
                first_beyond_life = current_tyre_life + (forecast_laps - need_forecast) + 1
                first_beyond_idx = first_beyond_life - 2
                steps_past_start = first_beyond_idx - len(history)
                total_steps = steps_past_start + need_forecast

                # VAR forecasts all compounds simultaneously
                forecast_matrix = self.var_model.forecast(
                    self.aligned_matrix[-self.var_model.k_ar:], steps=total_steps
                )
                forecast = forecast_matrix[:, comp_idx]
                forecast = np.maximum(forecast, 0.0)
                relevant = forecast[max(0, steps_past_start):]

                last_val = max(history[-1], 0.0)
                for val in relevant:
                    clamped = max(val, last_val)
                    result.append(clamped)
                    last_val = clamped
            elif need_forecast > 0:
                # No VAR model — simple linear extrapolation
                last_val = max(history[-1], 0.0)
                if len(history) > 1:
                    slope = max(history[-1] - history[-2], 0.001)
                else:
                    slope = 0.01
                for i in range(need_forecast):
                    last_val += slope
                    result.append(last_val)

            arr = np.array(result[:forecast_laps])
            arr = np.maximum(arr, 0.0)
            arr = np.minimum(arr, 5.0)
            return arr

        except Exception as e:
            logger.warning(f"VAR forecast failed for {compound}: {e}")
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

        For each stint the model's fitted curve prediction is compared
        against the stint's actual (clean) degradation at each lap.
        """
        all_actual = []
        all_predicted = []

        for stint_key, stint_df in ts_data.items():
            compound = stint_df["Compound"].iloc[0]
            if compound not in self.fitted_curves:
                continue

            fclt = stint_df["FuelCorrectedLapTime"].values
            deg = _clean_degradation_curve(fclt)
            if len(deg) < 5:
                continue

            curve = self.fitted_curves[compound]
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
        joblib.dump(self.history, os.path.join(path, "var_history.pkl"))
        joblib.dump(self.var_model, os.path.join(path, "var_model.pkl"))
        joblib.dump(self.aligned_matrix, os.path.join(path, "var_matrix.pkl"))
        joblib.dump(self.selected_lag, os.path.join(path, "var_lag.pkl"))
        joblib.dump(self.metrics, os.path.join(path, "var_metrics.pkl"))
        joblib.dump(self.fitted_curves, os.path.join(path, "var_fitted.pkl"))
        joblib.dump(self.diagnostics, os.path.join(path, "var_diagnostics.pkl"))
        logger.info(f"VAR saved to {path}")

    def load(self, path: str):
        """Load model artefacts from disk."""
        self.history = joblib.load(os.path.join(path, "var_history.pkl"))
        self.var_model = joblib.load(os.path.join(path, "var_model.pkl"))
        self.aligned_matrix = joblib.load(os.path.join(path, "var_matrix.pkl"))
        self.selected_lag = joblib.load(os.path.join(path, "var_lag.pkl"))
        metrics_path = os.path.join(path, "var_metrics.pkl")
        if os.path.exists(metrics_path):
            self.metrics = joblib.load(metrics_path)
        fitted_path = os.path.join(path, "var_fitted.pkl")
        if os.path.exists(fitted_path):
            self.fitted_curves = joblib.load(fitted_path)
        diag_path = os.path.join(path, "var_diagnostics.pkl")
        if os.path.exists(diag_path):
            self.diagnostics = joblib.load(diag_path)
        logger.info(f"VAR loaded from {path} — compounds: {list(self.history.keys())}")

    # ------------------------------------------------------------------
    # Strategy helper
    # ------------------------------------------------------------------
    def get_pit_recommendation(self, compound: str, current_tyre_life: int,
                                threshold: float = 1.5, max_forecast: int = 30) -> int:
        """Recommend when to pit based on degradation threshold."""
        forecast = self.predict_degradation(compound, current_tyre_life, max_forecast)
        for i, deg in enumerate(forecast):
            if deg > threshold:
                return i + 1
        return -1
