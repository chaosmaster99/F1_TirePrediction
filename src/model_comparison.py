"""
Model Comparison Module
Trains all models, evaluates them, and produces comparison metrics + visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import json

logger = logging.getLogger(__name__)

# Plot style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class ModelComparison:
    """
    Orchestrates training and comparison of all 5 models:
    1. ARIMA (Time Series)
    2. VAR — Vector Autoregression (Multivariate Time Series)
    3. XGBoost (ML)
    4. LSTM (DL)
    5. Hybrid XGBoost+LSTM (Ensemble)
    """
    
    def __init__(self, plots_dir: str = "plots", models_dir: str = "models"):
        self.plots_dir = plots_dir
        self.models_dir = models_dir
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        self.models = {}
        self.metrics = {}
        self.best_model_name = None
        self.best_model = None
    
    def train_all(self, ml_data: dict, lstm_data: dict, ts_data: dict):
        """Train all 5 models."""
        from src.models.arima_model import ARIMADegradationModel
        from src.models.var_model import VARDegradationModel
        from src.models.xgboost_model import XGBoostPitModel
        from src.models.lstm_model import LSTMDegradationModel
        from src.models.hybrid_model import HybridModel
        
        # 1. ARIMA
        logger.info("=" * 60)
        logger.info("Training Model 1/5: ARIMA (with stationarity diagnostics)")
        logger.info("=" * 60)
        arima = ARIMADegradationModel(order=(1, 1, 1))  # default fallback; auto-selects via ADF+AIC
        arima.fit(ts_data)
        self.models["ARIMA"] = arima
        
        # 2. VAR (Vector Autoregression)
        logger.info("=" * 60)
        logger.info("Training Model 2/5: VAR (Vector Autoregression)")
        logger.info("=" * 60)
        var = VARDegradationModel(maxlags=5)
        var.fit(ts_data)
        self.models["VAR"] = var
        
        # 3. XGBoost
        logger.info("=" * 60)
        logger.info("Training Model 3/5: XGBoost")
        logger.info("=" * 60)
        xgboost = XGBoostPitModel()
        xgboost.fit(ml_data)
        self.models["XGBoost"] = xgboost
        
        # 4. LSTM
        logger.info("=" * 60)
        logger.info("Training Model 4/5: LSTM")
        logger.info("=" * 60)
        lstm = LSTMDegradationModel(seq_length=8, epochs=80, batch_size=32)
        lstm.fit(lstm_data)
        self.models["LSTM"] = lstm
        
        # 5. Hybrid (ARIMA + XGBoost residual stacking)
        logger.info("=" * 60)
        logger.info("Training Model 5/5: Hybrid (ARIMA + XGBoost Residual)")
        logger.info("=" * 60)
        hybrid = HybridModel(arima)
        hybrid.fit(ml_data, ts_data)
        self.models["Hybrid"] = hybrid
        
        logger.info("All models trained successfully!")
    
    def evaluate_all(self, ml_data: dict, lstm_data: dict, ts_data: dict):
        """Evaluate all models and collect metrics."""
        logger.info("\nEvaluating all models...")
        
        # Time series models evaluate on ts_data
        for name in ["ARIMA", "VAR"]:
            if name in self.models:
                self.metrics[name] = self.models[name].evaluate(ts_data)
        
        # ML model evaluates on ml_data
        if "XGBoost" in self.models:
            self.metrics["XGBoost"] = self.models["XGBoost"].evaluate(ml_data)
        
        # LSTM evaluates on lstm_data
        if "LSTM" in self.models:
            self.metrics["LSTM"] = self.models["LSTM"].evaluate(lstm_data)
        
        # Hybrid evaluates on ml_data (instance-level, same as XGBoost)
        if "Hybrid" in self.models:
            self.metrics["Hybrid"] = self.models["Hybrid"].evaluate(ml_data=ml_data)
        
        # Find best model per evaluation level
        best_instance = (None, float("inf"))
        best_curve = (None, float("inf"))
        for name, met in self.metrics.items():
            rmse = met.get("rmse", float("inf"))
            lvl = self.EVAL_TYPE.get(name, "Unknown")
            if lvl == "Instance" and rmse < best_instance[1]:
                best_instance = (name, rmse)
            elif lvl == "Curve" and rmse < best_curve[1]:
                best_curve = (name, rmse)
        
        # Overall best = instance-level winner (larger test set, finer granularity)
        self.best_model_name = best_instance[0] or best_curve[0]
        self.best_model = self.models[self.best_model_name]
        self.best_instance = best_instance
        self.best_curve = best_curve
        
        logger.info(f"\nBest instance-level model: {best_instance[0]} (RMSE: {best_instance[1]:.4f})")
        logger.info(f"Best curve-level model:    {best_curve[0]} (RMSE: {best_curve[1]:.4f})")
        logger.info(f"Overall best: {self.best_model_name}")
        return self.metrics
    
    # Evaluation methodology for each model
    EVAL_TYPE = {
        "ARIMA":   "Curve",
        "VAR":     "Curve",
        "XGBoost": "Instance",
        "LSTM":    "Instance",
        "Hybrid":  "Instance",
    }

    def get_comparison_table(self) -> pd.DataFrame:
        """Create a comparison DataFrame."""
        rows = []
        for name, met in self.metrics.items():
            row = {"Model": name}
            row["Eval Level"] = self.EVAL_TYPE.get(name, "Unknown")
            row["RMSE"] = met.get("rmse", None)
            row["MAE"] = met.get("mae", None)
            row["R²"] = met.get("r2", None)
            
            # Classification metrics (only for XGBoost)
            if "classification" in met:
                row["Accuracy"] = met["classification"].get("accuracy", None)
                row["F1 Score"] = met["classification"].get("f1", None)
            else:
                row["Accuracy"] = None
                row["F1 Score"] = None
            
            rows.append(row)
        
        df = pd.DataFrame(rows).set_index("Model")
        return df
    
    def plot_comparison(self):
        """Generate comparison visualizations."""
        comp_table = self.get_comparison_table()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. RMSE Comparison
        rmse_vals = comp_table["RMSE"].dropna()
        colors = ["#2ecc71" if m == self.best_model_name else "#3498db" for m in rmse_vals.index]
        rmse_vals.plot(kind="bar", ax=axes[0], color=colors, edgecolor="black")
        axes[0].set_title("RMSE Comparison (Lower = Better)", fontsize=13, fontweight="bold")
        axes[0].set_ylabel("RMSE (seconds)")
        axes[0].tick_params(axis="x", rotation=45)
        for i, v in enumerate(rmse_vals.values):
            axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        
        # 2. MAE Comparison
        mae_vals = comp_table["MAE"].dropna()
        colors = ["#2ecc71" if m == self.best_model_name else "#e74c3c" for m in mae_vals.index]
        mae_vals.plot(kind="bar", ax=axes[1], color=colors, edgecolor="black")
        axes[1].set_title("MAE Comparison (Lower = Better)", fontsize=13, fontweight="bold")
        axes[1].set_ylabel("MAE (seconds)")
        axes[1].tick_params(axis="x", rotation=45)
        for i, v in enumerate(mae_vals.values):
            axes[1].text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        
        # 3. R² Comparison
        r2_vals = comp_table["R²"].dropna()
        colors = ["#2ecc71" if m == self.best_model_name else "#9b59b6" for m in r2_vals.index]
        r2_vals.plot(kind="bar", ax=axes[2], color=colors, edgecolor="black")
        axes[2].set_title("R² Comparison (Higher = Better)", fontsize=13, fontweight="bold")
        axes[2].set_ylabel("R² Score")
        axes[2].tick_params(axis="x", rotation=45)
        for i, v in enumerate(r2_vals.values):
            axes[2].text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, "model_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Comparison plot saved to {save_path}")
        
        return save_path
    
    def plot_degradation_curves(self, ts_data: dict, compound: str = "MEDIUM"):
        """Plot predicted degradation curves for all models that support it."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        tyre_life = 0  # Fresh tires
        forecast_laps = 35
        laps = np.arange(tyre_life + 1, tyre_life + forecast_laps + 1)
        
        # Distinct styles so overlapping curves stay visible
        styles = {
            "ARIMA":   {"color": "#E74C3C", "linestyle": "-",  "marker": "o", "markersize": 5, "linewidth": 3.0, "zorder": 4},
            "VAR":     {"color": "#F39C12", "linestyle": (0, (5, 5)), "marker": "s", "markersize": 7, "linewidth": 3.0, "zorder": 5},
            "Hybrid":  {"color": "#2ECC71", "linestyle": "-",  "marker": "^", "markersize": 4, "linewidth": 2.0, "zorder": 3},
            "XGBoost": {"color": "#3498DB", "linestyle": "-.", "marker": "D", "markersize": 4, "linewidth": 2.0, "zorder": 2},
            "LSTM":    {"color": "#9B59B6", "linestyle": ":",  "marker": "v", "markersize": 4, "linewidth": 2.0, "zorder": 1},
        }
        default_style = {"linestyle": "-", "marker": "o", "markersize": 3, "linewidth": 2}

        for name, model in self.models.items():
            style = styles.get(name, default_style)
            if hasattr(model, "predict_degradation"):
                try:
                    pred = model.predict_degradation(compound, tyre_life, forecast_laps)
                    ax.plot(laps, pred, label=name, **style)
                except:
                    pass
            elif hasattr(model, "predict_degradation_ts"):
                try:
                    pred = model.predict_degradation_ts(compound, tyre_life, forecast_laps)
                    ax.plot(laps, pred, label=name, **style)
                except:
                    pass
        
        ax.set_xlabel("Lap Number", fontsize=12)
        ax.set_ylabel("Predicted Degradation (seconds)", fontsize=12)
        ax.set_title(f"Tire Degradation Forecast - {compound} Compound", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.plots_dir, f"degradation_curves_{compound.lower()}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Degradation curves saved to {save_path}")
        
        return save_path

    def save_degradation_data(self):
        """Save degradation prediction data as JSON for interactive frontend plots."""
        data = {}
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            data[compound] = {}
            tyre_life = 0
            forecast_laps = 35
            laps = list(range(tyre_life + 1, tyre_life + forecast_laps + 1))
            data[compound]["laps"] = laps
            for name, model in self.models.items():
                try:
                    pred = model.predict_degradation(compound, tyre_life, forecast_laps)
                    data[compound][name] = [round(float(v), 4) for v in pred]
                except Exception:
                    pass
        
        path = os.path.join(self.plots_dir, "degradation_data.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Degradation prediction data saved to {path}")
    
    def plot_feature_importance(self):
        """Plot XGBoost feature importance."""
        if "XGBoost" not in self.models:
            return None
        
        importance = self.models["XGBoost"].feature_importance
        if not importance:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_imp = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        ax.barh(list(sorted_imp.keys()), list(sorted_imp.values()), color="#3498db", edgecolor="black")
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_title("XGBoost Feature Importance", fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        
        save_path = os.path.join(self.plots_dir, "feature_importance.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Feature importance plot saved to {save_path}")
        
        return save_path
    
    def save_results(self):
        """Save all comparison results."""
        # Save metrics
        metrics_serializable = {}
        for name, met in self.metrics.items():
            metrics_serializable[name] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in met.items()
                if not isinstance(v, dict)
            }
            # Handle nested dicts
            for k, v in met.items():
                if isinstance(v, dict):
                    metrics_serializable[name][k] = {
                        kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                        for kk, vv in v.items()
                    }
            # Add evaluation type
            metrics_serializable[name]["eval_level"] = self.EVAL_TYPE.get(name, "Unknown")
        
        # No extra hybrid metadata to serialize for residual stacking
        
        with open(os.path.join(self.plots_dir, "metrics.json"), "w") as f:
            json.dump(metrics_serializable, f, indent=2)
        
        # Save comparison table
        comp_table = self.get_comparison_table()
        comp_table.to_csv(os.path.join(self.plots_dir, "comparison_table.csv"))
        
        # Save best model info
        best_info = {
            "best_model": self.best_model_name,
            "best_instance": {
                "name": self.best_instance[0],
                "rmse": self.best_instance[1],
            } if hasattr(self, "best_instance") and self.best_instance[0] else {},
            "best_curve": {
                "name": self.best_curve[0],
                "rmse": self.best_curve[1],
            } if hasattr(self, "best_curve") and self.best_curve[0] else {},
            "metrics": metrics_serializable.get(self.best_model_name, {}),
        }
        with open(os.path.join(self.models_dir, "best_model_info.json"), "w") as f:
            json.dump(best_info, f, indent=2)
        
        # Save individual models
        for name, model in self.models.items():
            if hasattr(model, "save"):
                model_dir = os.path.join(self.models_dir, name.lower().replace(" ", "_"))
                try:
                    model.save(model_dir)
                except Exception as e:
                    logger.warning(f"Could not save {name}: {e}")
        
        logger.info(f"Results saved to {self.plots_dir}/ and {self.models_dir}/")
    
    def generate_full_report(self) -> str:
        """Generate a text report of all results."""
        report = []
        report.append("=" * 70)
        report.append("F1 PIT STOP PREDICTION - MODEL COMPARISON REPORT")
        report.append("=" * 70)
        report.append("")
        
        comp_table = self.get_comparison_table()
        report.append("MODEL PERFORMANCE COMPARISON:")
        report.append("-" * 50)
        report.append(comp_table.to_string())
        report.append("")
        
        report.append("EVALUATION METHODOLOGY:")
        report.append("-" * 50)
        report.append("Instance-level: model predicts per lap-row, compared to actual")
        report.append("  lap time delta. Larger test set, finer granularity.")
        report.append("Curve-level: model forecasts degradation curves per stint, compared")
        report.append("  to averaged actual degradation. Comparable across TS models.")
        report.append("")
        report.append("NOTE: Instance and Curve metrics are NOT directly comparable.")
        report.append("      XGBoost/LSTM instance RMSE << ARIMA/VAR/Hybrid curve RMSE")
        report.append("      is expected due to different evaluation granularity.")
        report.append("")
        
        bi = getattr(self, "best_instance", (self.best_model_name, 0))
        bc = getattr(self, "best_curve", (None, 0))
        report.append(f"BEST INSTANCE-LEVEL MODEL: {bi[0]}")
        if bi[0]:
            m = self.metrics[bi[0]]
            report.append(f"  RMSE: {m.get('rmse', 0):.4f}  MAE: {m.get('mae', 0):.4f}  R2: {m.get('r2', 0):.4f}")
        report.append(f"BEST CURVE-LEVEL MODEL:    {bc[0]}")
        if bc[0]:
            m = self.metrics[bc[0]]
            report.append(f"  RMSE: {m.get('rmse', 0):.4f}  MAE: {m.get('mae', 0):.4f}  R2: {m.get('r2', 0):.4f}")
        report.append("")
        
        # Hybrid model info
        if "Hybrid" in self.models:
            report.append("HYBRID MODEL (Residual Stacking):")
            report.append("-" * 50)
            report.append("  Base: ARIMA degradation curve per compound")
            report.append("  Correction: XGBoost trained on residuals")
            report.append("  Final = ARIMA_prediction + XGBoost_residual")
            report.append("")
        
        # XGBoost classification
        if "XGBoost" in self.metrics and "classification" in self.metrics["XGBoost"]:
            cls = self.metrics["XGBoost"]["classification"]
            report.append("XGBOOST PIT STOP CLASSIFICATION:")
            report.append("-" * 50)
            report.append(f"  Accuracy:  {cls.get('accuracy', 0):.4f}")
            report.append(f"  Precision: {cls.get('precision', 0):.4f}")
            report.append(f"  Recall:    {cls.get('recall', 0):.4f}")
            report.append(f"  F1 Score:  {cls.get('f1', 0):.4f}")
            report.append("")
        
        report.append("MODEL DESCRIPTIONS:")
        report.append("-" * 50)
        report.append("1. ARIMA: Autoregressive Integrated Moving Average [Curve]")
        report.append("   - Time series model on averaged per-compound degradation curves")
        report.append("   - Models degradation as an autoregressive process")
        report.append("")
        report.append("2. VAR: Vector Autoregression [Curve]")
        report.append("   - Multivariate model treating all compound curves as a system")
        report.append("   - Captures cross-compound dynamics and lead-lag relationships")
        report.append("")
        report.append("3. XGBoost: Gradient Boosted Decision Trees [Instance]")
        report.append("   - Feature-based ML model capturing non-linear interactions")
        report.append("   - Considers compound, circuit, driver effects simultaneously")
        report.append("   - Also includes pit stop binary classifier (F1 score)")
        report.append("")
        report.append("4. LSTM: Long Short-Term Memory Neural Network [Instance]")
        report.append("   - Deep learning model for sequential data")
        report.append("   - Autoregressive per-step prediction")
        report.append("")
        report.append("5. Hybrid (ARIMA+XGBoost): Residual Stacking [Instance]")
        report.append("   - ARIMA base prediction + XGBoost residual correction")
        report.append("   - Combines time-series structure with feature-based learning")
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        with open(os.path.join(self.plots_dir, "comparison_report.txt"), "w",
                  encoding="utf-8") as f:
            f.write(report_text)
        
        return report_text
