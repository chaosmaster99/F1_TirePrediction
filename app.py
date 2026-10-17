"""
F1 Pit Stop Strategy Predictor - Streamlit Frontend
Interactive web app for predicting optimal pit stop timing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import json
import sys
import logging

# Suppress warnings and TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.strategy import PitStopStrategy, RACE_LAPS, DEFAULT_THRESHOLDS, PIT_STOP_LOSS, DRIVER_PROFILES, DRIVER_ABBREV

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="F1 Pit Stop Strategy Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #e10600 0%, #1e1e1e 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
    }
    .main-header p {
        color: #cccccc;
        font-size: 1.1rem;
    }
    .stint-card {
        background: #1e1e1e;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .stint-soft { border-left-color: #ff0000; }
    .stint-medium { border-left-color: #ffd700; }
    .stint-hard { border-left-color: #ffffff; }
    .metric-card {
        background: linear-gradient(135deg, #2d2d2d, #1a1a1a);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #333;
    }
    .stButton > button {
        background-color: #e10600;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #ff1a1a;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Initialize Strategy Engine
# ============================================================
@st.cache_resource
def load_strategy_engine():
    """Load models and create strategy engine."""
    engine = PitStopStrategy(models_dir="models")
    
    # Try to load trained models
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    
    best_model_path = os.path.join(models_dir, "best_model_info.json")
    metrics_path = os.path.join(plots_dir, "metrics.json")
    
    has_trained_models = False
    metrics = {}
    best_model_name = "Analytical (Fallback)"
    
    if os.path.exists(best_model_path):
        with open(best_model_path) as f:
            info = json.load(f)
            best_model_name = info.get("best_model", "Unknown")
            has_trained_models = True
    
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    
    # Try to load actual model objects
    try:
        from src.models.arima_model import ARIMADegradationModel
        from src.models.var_model import VARDegradationModel
        from src.models.lstm_model import LSTMDegradationModel
        
        # Load ARIMA model if saved
        arima_dir = os.path.join(models_dir, "arima")
        if os.path.exists(os.path.join(arima_dir, "arima_history.pkl")):
            arima = ARIMADegradationModel()
            arima.load(arima_dir)
            engine.arima_model = arima

        # Load VAR model if saved
        var_dir = os.path.join(models_dir, "var")
        if os.path.exists(os.path.join(var_dir, "var_history.pkl")):
            var = VARDegradationModel()
            var.load(var_dir)
            engine.var_model = var

        # Load LSTM model if saved
        lstm_dir = os.path.join(models_dir, "lstm")
        if os.path.exists(os.path.join(lstm_dir, "lstm_model.keras")):
            lstm = LSTMDegradationModel()
            lstm.load(lstm_dir)
            engine.lstm_model = lstm
    except Exception:
        pass

    # Load XGBoost model + preprocessing artifacts (scaler, encoders, baselines)
    engine.load_models()

    # Load Hybrid model if ARIMA is available (residual stacking)
    try:
        from src.models.hybrid_model import HybridModel
        hybrid_dir = os.path.join(models_dir, "hybrid")
        if (os.path.exists(os.path.join(hybrid_dir, "hybrid_config.pkl"))
                and engine.arima_model is not None):
            hybrid = HybridModel(engine.arima_model)
            hybrid.load(hybrid_dir)
            engine.hybrid_model = hybrid
    except Exception:
        pass

    if engine._xgb_ready:
        best_model_name = "XGBoost (Trained)"
    
    return engine, has_trained_models, metrics, best_model_name


@st.cache_data
def load_comparison_data():
    """Load comparison table, plots, and degradation data."""
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    
    comp_table = None
    comp_path = os.path.join(plots_dir, "comparison_table.csv")
    if os.path.exists(comp_path):
        comp_table = pd.read_csv(comp_path, index_col=0)
    
    report = ""
    report_path = os.path.join(plots_dir, "comparison_report.txt")
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
    
    deg_data = {}
    deg_path = os.path.join(plots_dir, "degradation_data.json")
    if os.path.exists(deg_path):
        with open(deg_path) as f:
            deg_data = json.load(f)
    
    return comp_table, report, deg_data


# ============================================================
# Compound colors
# ============================================================
COMPOUND_COLORS = {
    "SOFT": "#FF0000",
    "MEDIUM": "#FFD700",
    "HARD": "#FFFFFF",
}


def get_compound_emoji(compound):
    return {"SOFT": "🔴", "MEDIUM": "🟡", "HARD": "⚪"}.get(compound, "🔵")


# ============================================================
# Circuits and Drivers
# ============================================================
CIRCUITS = sorted(RACE_LAPS.keys())

# Known F1 drivers (with historical degradation profiles)
DRIVERS = [
    "Max Verstappen", "Lewis Hamilton", "Charles Leclerc",
    "Lando Norris", "Carlos Sainz",
]


# ============================================================
# Main App
# ============================================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏎️ F1 Pit Stop Strategy Predictor</h1>
        <p>Tire Degradation Analysis & Optimal Pit Stop Timing</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load resources
    engine, has_models, metrics, best_model_name = load_strategy_engine()
    comp_table, report, deg_data = load_comparison_data()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🏁 Race Configuration")
        st.markdown("---")
        
        # Driver selection
        driver = st.selectbox(
            "👤 Select Driver",
            DRIVERS,
            index=0,
        )        
        # Show driver profile info
        driver_code = DRIVER_ABBREV.get(driver)
        if driver_code and driver_code in DRIVER_PROFILES:
            profile = DRIVER_PROFILES[driver_code]
            st.caption(f"Team: {profile['team']} | Style: {profile['style'].title()}")        
        # Circuit selection
        circuit = st.selectbox(
            "🏟️ Select Circuit",
            CIRCUITS,
            index=CIRCUITS.index("Melbourne") if "Melbourne" in CIRCUITS else 0,
        )
        
        total_laps = RACE_LAPS.get(circuit, 56)
        st.info(f"📏 Total Race Laps: **{total_laps}**")
        
        # Starting compound
        starting_compound = st.selectbox(
            "🛞 Starting Tire Compound",
            ["SOFT", "MEDIUM", "HARD"],
            index=1,
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Advanced Settings")
        
        # Threshold
        threshold = st.slider(
            "Degradation Threshold (seconds)",
            min_value=0.5,
            max_value=3.0,
            value=DEFAULT_THRESHOLDS.get(starting_compound, 1.5),
            step=0.1,
            help="Threshold for the starting compound. Other compounds scale proportionally (e.g. HARD uses a higher threshold than SOFT)."
        )
        
        st.markdown("---")
        
        # Model info
        st.markdown("### 🤖 Model Info")
        if has_models:
            st.success(f"Active: **{best_model_name}**")
        else:
            st.warning("Using analytical fallback model. Run pipeline for ML models.")
        
        st.markdown("---")
        predict_button = st.button("🏁 Generate Strategy", use_container_width=True)  # noqa
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Strategy Prediction",
        "📈 Model Comparison",
        "🔍 Degradation Analysis",
        "📉 ARIMA Diagnostics",
        "📋 About"
    ])
    
    # ============================================================
    # Tab 1: Strategy Prediction
    # ============================================================
    with tab1:
        if predict_button or "last_strategy" in st.session_state:
            if predict_button:
                # Generate strategy
                with st.spinner("Calculating optimal strategy..."):
                    strategy = engine.generate_strategy(
                        starting_compound=starting_compound,
                        total_race_laps=total_laps,
                        circuit=circuit,
                        driver=driver,
                        threshold=threshold,
                    )
                    
                    # Also get all starting compound options
                    all_strategies = engine.compare_strategies(
                        circuit=circuit,
                        total_laps=total_laps,
                        driver=driver,
                        threshold=threshold,
                    )
                    
                    st.session_state["last_strategy"] = strategy
                    st.session_state["all_strategies"] = all_strategies
            else:
                strategy = st.session_state.get("last_strategy")
                all_strategies = st.session_state.get("all_strategies", [])
            
            if strategy:
                # Top metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🔧 Pit Stops", strategy["total_pit_stops"])
                with col2:
                    st.metric("⏱️ Pit Time Loss", f"{strategy['estimated_total_pit_time_loss']:.0f}s")
                with col3:
                    st.metric("📉 Tire Deg Loss", f"{strategy['estimated_total_degradation_loss']:.1f}s")
                with col4:
                    st.metric("📊 Total Impact", f"{strategy['estimated_total_time_impact']:.1f}s")
                
                st.markdown("---")
                
                # Strategy visualization
                st.markdown("### 🏁 Recommended Race Strategy")
                
                # Create stint visual
                fig = go.Figure()
                
                for stint in strategy["stints"]:
                    compound = stint["compound"]
                    color = COMPOUND_COLORS.get(compound, "#888888")
                    
                    fig.add_trace(go.Bar(
                        name=f"Stint {stint['stint_number']}: {compound}",
                        x=[stint["stint_length"]],
                        y=[f"Strategy"],
                        orientation="h",
                        marker=dict(
                            color=color,
                            line=dict(color="black", width=1)
                        ),
                        text=f"{compound}<br>Laps {stint['start_lap']}-{stint['end_lap']}<br>({stint['stint_length']} laps)",
                        textposition="inside",
                        textfont=dict(color="black" if compound != "HARD" else "black", size=12),
                        hovertemplate=(
                            f"<b>Stint {stint['stint_number']}</b><br>"
                            f"Compound: {compound}<br>"
                            f"Laps: {stint['start_lap']} - {stint['end_lap']}<br>"
                            f"Length: {stint['stint_length']} laps<br>"
                            f"Avg Degradation: {stint['avg_degradation']:.3f}s<br>"
                            f"Max Degradation: {stint['max_degradation']:.3f}s<br>"
                            f"<extra></extra>"
                        ),
                    ))
                
                fig.update_layout(
                    barmode="stack",
                    height=150,
                    margin=dict(l=0, r=0, t=30, b=0),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    xaxis_title="Laps",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                
                st.plotly_chart(fig, width="stretch")
                
                # Stint details
                st.markdown("### 📋 Stint Details")
                
                stint_cols = st.columns(len(strategy["stints"]))
                for i, stint in enumerate(strategy["stints"]):
                    with stint_cols[i]:
                        emoji = get_compound_emoji(stint["compound"])
                        st.markdown(f"#### {emoji} Stint {stint['stint_number']}")
                        st.markdown(f"**Compound:** {stint['compound']}")
                        st.markdown(f"**Laps:** {stint['start_lap']} → {stint['end_lap']}")
                        st.markdown(f"**Length:** {stint['stint_length']} laps")
                        st.markdown(f"**Tire Age at End:** {stint['tyre_life_at_end']} laps")
                        st.markdown(f"**Avg Deg:** {stint['avg_degradation']:.3f}s")
                        st.markdown(f"**Max Deg:** {stint['max_degradation']:.3f}s")
                        
                        if i < len(strategy["stints"]) - 1:
                            st.markdown(f"**→ Pit on Lap {stint['end_lap']}**")
                
                # Degradation curves per stint
                st.markdown("### 📈 Degradation Curves")
                
                fig_deg = go.Figure()
                for stint in strategy["stints"]:
                    deg_curve = stint["degradation_curve"]
                    if isinstance(deg_curve, list) and len(deg_curve) > 0:
                        laps = list(range(stint["start_lap"], stint["start_lap"] + len(deg_curve)))
                        color = COMPOUND_COLORS.get(stint["compound"], "#888888")
                        
                        fig_deg.add_trace(go.Scatter(
                            x=laps,
                            y=deg_curve,
                            mode="lines+markers",
                            name=f"Stint {stint['stint_number']}: {stint['compound']}",
                            line=dict(color=color, width=2),
                            marker=dict(size=4),
                        ))
                
                # Add threshold line
                fig_deg.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Pit Threshold ({threshold}s)",
                    annotation_position="top right",
                )
                
                fig_deg.update_layout(
                    title="Tire Degradation Over Race Distance",
                    xaxis_title="Lap Number",
                    yaxis_title="Degradation (seconds)",
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                
                st.plotly_chart(fig_deg, width="stretch")
                
                # Alternative strategies comparison
                if all_strategies and len(all_strategies) > 1:
                    st.markdown("### 🔄 Strategy Comparison (All Starting Compounds)")
                    
                    strat_data = []
                    for s in all_strategies:
                        strat_data.append({
                            "Starting Compound": s["stints"][0]["compound"],
                            "Pit Stops": s["total_pit_stops"],
                            "Compounds Used": ", ".join(s["compounds_used"]),
                            "Degradation Loss (s)": round(s["estimated_total_degradation_loss"], 1),
                            "Pit Time Loss (s)": round(s["estimated_total_pit_time_loss"], 1),
                            "Total Impact (s)": round(s["estimated_total_time_impact"], 1),
                            "Rank": s.get("rank", ""),
                            "Optimal": "✅" if s.get("is_optimal", False) else "",
                        })
                    
                    st.dataframe(
                        pd.DataFrame(strat_data),
                        width="stretch",
                        hide_index=True,
                    )
        else:
            # Welcome state
            st.markdown("### 👈 Configure your race and click **Generate Strategy**")
            st.markdown("""
            **How to use:**
            1. Select a **driver** and **circuit** from the sidebar
            2. Choose the **starting tire compound**
            3. Adjust the **degradation threshold** if needed
            4. Click **Generate Strategy** to get pit stop recommendations
            
            **For the Australian GP (March 8):**
            - Select **Melbourne** as the circuit (58 laps)
            - The model will predict optimal pit windows based on tire degradation analysis
            """)
            
            # Show quick stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏎️ Circuits Available", len(CIRCUITS))
            with col2:
                st.metric("👤 Drivers", len(DRIVERS))
            with col3:
                st.metric("🤖 Models Trained", "5" if has_models else "0 (Fallback)")
    
    # ============================================================
    # Tab 2: Model Comparison
    # ============================================================
    with tab2:
        st.markdown("### Model Performance Comparison")
        
        if comp_table is not None:
            # Methodology explanation
            st.info(
                "**Evaluation Levels:** Instance-level models (XGBoost, LSTM, Hybrid) are evaluated per-lap against "
                "individual data points. Curve-level models (ARIMA, VAR) are evaluated against "
                "averaged degradation curves. Metrics across levels are **not directly comparable**."
            )
            
            # Format the table display
            numeric_cols = ["RMSE", "MAE", "R²", "Accuracy", "F1 Score"]
            fmt = {c: "{:.4f}" for c in numeric_cols if c in comp_table.columns}
            styled = comp_table.style.format(fmt, na_rep="—")
            if "RMSE" in comp_table.columns and "MAE" in comp_table.columns:
                styled = styled.highlight_min(subset=["RMSE", "MAE"], color="#2ecc71")
            if "R²" in comp_table.columns:
                styled = styled.highlight_max(subset=["R²"], color="#2ecc71")
            st.dataframe(styled, width="stretch")

            # Best model cards
            models_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            best_info_path = os.path.join(models_dir_path, "best_model_info.json")
            if os.path.exists(best_info_path):
                with open(best_info_path) as f:
                    best_info = json.load(f)
                bi = best_info.get("best_instance", {})
                bc = best_info.get("best_curve", {})
                col_bi, col_bc = st.columns(2)
                with col_bi:
                    if bi.get("name"):
                        st.success(f"**Best Instance-Level:** {bi['name']} (RMSE: {bi.get('rmse', 0):.4f})")
                with col_bc:
                    if bc.get("name"):
                        st.success(f"**Best Curve-Level:** {bc['name']} (RMSE: {bc.get('rmse', 0):.4f})")
            
            # RMSE / MAE grouped bar chart
            models_list = comp_table.index.tolist()
            eval_levels = comp_table["Eval Level"].tolist() if "Eval Level" in comp_table.columns else [""] * len(models_list)
            bar_colors_rmse = ["#e74c3c" if lv == "Instance" else "#c0392b" for lv in eval_levels]
            bar_colors_mae = ["#3498db" if lv == "Instance" else "#2980b9" for lv in eval_levels]
            
            fig_metrics = go.Figure()
            fig_metrics.add_trace(go.Bar(
                name="RMSE",
                x=models_list,
                y=comp_table["RMSE"].values,
                marker_color=bar_colors_rmse,
                text=[f"{v:.4f}" for v in comp_table["RMSE"].values],
                textposition="auto",
            ))
            fig_metrics.add_trace(go.Bar(
                name="MAE",
                x=models_list,
                y=comp_table["MAE"].values,
                marker_color=bar_colors_mae,
                text=[f"{v:.4f}" for v in comp_table["MAE"].values],
                textposition="auto",
            ))
            fig_metrics.update_layout(
                title="Model Error Metrics (Lower = Better)",
                barmode="group",
                height=420,
                yaxis_title="Error (seconds)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_metrics, width="stretch")
            
            # R² chart
            r2_vals = comp_table["R²"].values
            r2_colors = ["#2ecc71" if v == max(r2_vals) else "#9b59b6" for v in r2_vals]
            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Bar(
                x=models_list,
                y=r2_vals,
                marker_color=r2_colors,
                text=[f"{v:.4f}" for v in r2_vals],
                textposition="auto",
            ))
            fig_r2.update_layout(
                title="R² Score (Higher = Better)",
                height=360,
                yaxis_title="R² Score",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_r2, width="stretch")
            
            # XGBoost classification metrics
            plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
            metrics_path = os.path.join(plots_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    all_metrics = json.load(f)
                xgb_cls = all_metrics.get("XGBoost", {}).get("classification", {})
                if xgb_cls:
                    st.markdown("#### XGBoost Pit-Stop Classification")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy", f"{xgb_cls.get('accuracy', 0):.2%}")
                    c2.metric("Precision", f"{xgb_cls.get('precision', 0):.2%}")
                    c3.metric("Recall", f"{xgb_cls.get('recall', 0):.2%}")
                    c4.metric("F1 Score", f"{xgb_cls.get('f1', 0):.2%}")
                
                # Hybrid residual stacking info
                hybrid_metrics = all_metrics.get("Hybrid", {})
                if hybrid_metrics.get("r2") is not None:
                    st.markdown("#### Hybrid Model — Residual Stacking (ARIMA + XGBoost)")
                    hc1, hc2, hc3 = st.columns(3)
                    hc1.metric("Base Model", "ARIMA", help="Per-compound degradation curve")
                    hc2.metric("Residual Corrector", "XGBoost", help="Trained on Actual − ARIMA residuals")
                    hc3.metric("Hybrid R²", f"{hybrid_metrics['r2']:.4f}")
            
            # Feature importance image
            feature_imp_path = os.path.join(plots_dir, "feature_importance.png")
            if os.path.exists(feature_imp_path):
                st.markdown("#### XGBoost Feature Importance")
                st.image(feature_imp_path)
            
            # Interactive degradation curve overlay (all models per compound)
            if deg_data:
                st.markdown("#### Degradation Curve Overlay (All Models)")
                overlay_compound = st.selectbox(
                    "Select Compound",
                    [c for c in ["SOFT", "MEDIUM", "HARD"] if c in deg_data],
                    key="overlay_compound",
                )
                if overlay_compound and overlay_compound in deg_data:
                    cd = deg_data[overlay_compound]
                    laps = cd.get("laps", [])
                    model_colors = {
                        "ARIMA": "#E74C3C", "VAR": "#F39C12", "XGBoost": "#3498DB",
                        "LSTM": "#9B59B6", "Hybrid": "#2ECC71",
                    }
                    model_dashes = {
                        "ARIMA": "solid", "VAR": "dash", "XGBoost": "dashdot",
                        "LSTM": "dot", "Hybrid": "solid",
                    }
                    fig_overlay = go.Figure()
                    for mname in ["ARIMA", "VAR", "XGBoost", "LSTM", "Hybrid"]:
                        if mname in cd:
                            fig_overlay.add_trace(go.Scatter(
                                x=laps, y=cd[mname],
                                mode="lines+markers",
                                name=mname,
                                line=dict(
                                    color=model_colors.get(mname, "#888"),
                                    width=2.5,
                                    dash=model_dashes.get(mname, "solid"),
                                ),
                                marker=dict(size=3),
                            ))
                    fig_overlay.update_layout(
                        title=f"All Models - {overlay_compound} Degradation Prediction",
                        xaxis_title="Tire Age (laps)",
                        yaxis_title="Predicted Degradation (seconds)",
                        height=480,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    )
                    st.plotly_chart(fig_overlay, width="stretch")
            
        else:
            st.warning("No model comparison data available. Run the pipeline first:")
            st.code("python run_pipeline.py", language="bash")
        
        # Full report
        if report:
            with st.expander("Full Comparison Report"):
                st.text(report)
    
    # ============================================================
    # Tab 3: Degradation Analysis
    # ============================================================
    with tab3:
        st.markdown("### Tire Degradation Deep Dive")
        
        analysis_compound = st.selectbox(
            "Select Compound for Analysis",
            ["SOFT", "MEDIUM", "HARD"],
            key="analysis_compound",
        )
        
        max_laps = st.slider("Forecast Laps", 10, 50, 30, key="analysis_laps")
        
        # Driver comparison for this compound
        st.markdown("### Driver Comparison")
        
        driver_colors = {
            "Max Verstappen": "#3671C6",   # Red Bull blue
            "Lewis Hamilton": "#E8002D",    # Ferrari red
            "Charles Leclerc": "#E8002D",   # Ferrari red
            "Lando Norris": "#FF8000",      # McLaren orange
            "Carlos Sainz": "#1868DB",      # Williams blue
        }
        driver_dashes = {
            "Max Verstappen": "solid",
            "Lewis Hamilton": "solid",
            "Charles Leclerc": "dash",
            "Lando Norris": "solid",
            "Carlos Sainz": "solid",
        }
        
        fig_drivers = go.Figure()
        for drv_name in DRIVERS:
            deg_curve = engine.predict_degradation_curve(
                analysis_compound, max_laps, 0, driver=drv_name, circuit=circuit
            )
            laps = list(range(1, len(deg_curve) + 1))
            fig_drivers.add_trace(go.Scatter(
                x=laps,
                y=deg_curve,
                mode="lines",
                name=drv_name,
                line=dict(
                    color=driver_colors.get(drv_name, "#888"),
                    width=2.5,
                    dash=driver_dashes.get(drv_name, "solid"),
                ),
            ))
        
        # Threshold line for selected compound
        thresh = DEFAULT_THRESHOLDS.get(analysis_compound, 1.5)
        fig_drivers.add_hline(
            y=thresh,
            line_dash="dot",
            line_color="red",
            annotation_text=f"Pit threshold ({thresh}s)",
            annotation_position="top right",
        )
        
        fig_drivers.update_layout(
            title=f"Driver Degradation Comparison - {analysis_compound} Compound",
            xaxis_title="Tire Age (laps)",
            yaxis_title="Degradation (seconds from fresh pace)",
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        
        st.plotly_chart(fig_drivers, width="stretch")
        
        # Compound comparison (using selected driver from sidebar)
        st.markdown(f"### Compound Comparison for {driver}")
        
        fig_comp = go.Figure()
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            deg = engine.predict_degradation_curve(compound, 40, 0, driver=driver, circuit=circuit)
            laps = list(range(1, len(deg) + 1))
            fig_comp.add_trace(go.Scatter(
                x=laps,
                y=deg,
                mode="lines",
                name=compound,
                line=dict(
                    color=COMPOUND_COLORS[compound],
                    width=3,
                ),
                fill="tozeroy",
                fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(COMPOUND_COLORS[compound])) + [0.1])}",
            ))
        
        fig_comp.update_layout(
            title="All Compounds - Degradation Over Tire Life",
            xaxis_title="Tire Age (laps)",
            yaxis_title="Degradation (seconds)",
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        
        st.plotly_chart(fig_comp, width="stretch")
        
        # Interactive model predictions overlay for selected compound
        if deg_data and analysis_compound in deg_data:
            st.markdown(f"### Model Predictions - {analysis_compound}")
            cd = deg_data[analysis_compound]
            laps = cd.get("laps", [])
            model_colors = {
                "ARIMA": "#E74C3C", "VAR": "#F39C12", "XGBoost": "#3498DB",
                "LSTM": "#9B59B6", "Hybrid": "#2ECC71",
            }
            model_dashes = {
                "ARIMA": "solid", "VAR": "dash", "XGBoost": "dashdot",
                "LSTM": "dot", "Hybrid": "solid",
            }
            fig_model_pred = go.Figure()
            for mname in ["ARIMA", "VAR", "XGBoost", "LSTM", "Hybrid"]:
                if mname in cd:
                    fig_model_pred.add_trace(go.Scatter(
                        x=laps, y=cd[mname],
                        mode="lines+markers",
                        name=mname,
                        line=dict(
                            color=model_colors.get(mname, "#888"),
                            width=2.5,
                            dash=model_dashes.get(mname, "solid"),
                        ),
                        marker=dict(size=3),
                    ))
            # Threshold line
            thresh = DEFAULT_THRESHOLDS.get(analysis_compound, 1.5)
            fig_model_pred.add_hline(
                y=thresh, line_dash="dot", line_color="red",
                annotation_text=f"Pit threshold ({thresh}s)",
                annotation_position="top right",
            )
            fig_model_pred.update_layout(
                title=f"All Models - {analysis_compound} Degradation Forecast",
                xaxis_title="Tire Age (laps)",
                yaxis_title="Predicted Degradation (seconds)",
                height=480,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_model_pred, width="stretch")
    
    # ============================================================
    # Tab 4: ARIMA Diagnostics
    # ============================================================
    with tab4:
        st.markdown("### ARIMA Time Series Diagnostics")
        st.info(
            "**Proper TS workflow:** For each compound the model runs ADF & KPSS "
            "stationarity tests, determines differencing order *d*, analyses ACF/PACF "
            "to guide *(p, q)* selection, then picks the best ARIMA(p,d,q) via AIC grid search. "
            "Residual diagnostics (Ljung-Box) validate the fit."
        )

        # Load diagnostics JSON
        diag_json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "plots", "arima_diagnostics", "diagnostics.json"
        )
        arima_diag_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "plots", "arima_diagnostics"
        )

        if os.path.exists(diag_json_path):
            with open(diag_json_path) as f:
                arima_diag = json.load(f)

            # Compound selector
            diag_compounds = list(arima_diag.keys())
            diag_compound = st.selectbox(
                "Select Compound", diag_compounds, key="diag_compound"
            )

            if diag_compound and diag_compound in arima_diag:
                cd = arima_diag[diag_compound]

                # --- Summary metrics ---
                st.markdown(f"#### {diag_compound} — Diagnostic Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    adf_p = cd.get("adf_original", {}).get("p_value", None)
                    st.metric("ADF p-value (original)", f"{adf_p:.4f}" if adf_p is not None else "N/A")
                    st.caption("❌ Non-stationary (expected — degradation has a trend)")
                with col2:
                    after_diff = cd.get("stationary_after_diff",
                                        cd.get("adf_differenced", {}).get("is_stationary", False))
                    diff_icon = "✅" if after_diff else "❌"
                    adf_diff_p = cd.get("adf_differenced", {}).get("p_value", None)
                    st.metric("After Differencing", f"{diff_icon} {'Stationary' if after_diff else 'Non-stationary'}")
                    st.caption(f"ADF p={adf_diff_p:.4f}" if adf_diff_p is not None else "")
                with col3:
                    d_val = cd.get("d_selected", "?")
                    order = cd.get("selected_order", [])
                    st.metric("Differencing (d)", str(d_val))
                    st.caption(f"Auto-selected order: ARIMA{tuple(order)}")
                with col4:
                    aic = cd.get("best_aic", None)
                    st.metric("Best AIC", f"{aic:.2f}" if aic is not None else "N/A")
                    series_len = cd.get("series_length", "?")
                    st.caption(f"Curve length: {series_len} laps")

                # --- ADF details ---
                with st.expander("ADF Test Details"):
                    adf_orig = cd.get("adf_original", {})
                    adf_diff = cd.get("adf_differenced", {})
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Original Series**")
                        st.json(adf_orig)
                    with c2:
                        st.markdown(f"**After Differencing (d={cd.get('d_selected', '?')})**")
                        st.json(adf_diff)

                # --- Ljung-Box residual test ---
                lb = cd.get("ljung_box", {})
                if lb:
                    with st.expander("Ljung-Box Residual Test"):
                        lb_rows = []
                        for lag, vals in lb.items():
                            lb_rows.append({
                                "Lag": int(lag),
                                "Statistic": round(vals["statistic"], 4),
                                "p-value": round(vals["p_value"], 4),
                                "White Noise?": "✅ Yes" if vals["p_value"] > 0.05 else "❌ No",
                            })
                        st.dataframe(pd.DataFrame(lb_rows), hide_index=True)
                        st.caption("p > 0.05 → residuals are white noise (good fit)")

                st.markdown("---")

                # --- Stationarity plot ---
                st.markdown(f"#### Stationarity Analysis — {diag_compound}")
                stat_img = os.path.join(arima_diag_dir, f"stationarity_{diag_compound.lower()}.png")
                if os.path.exists(stat_img):
                    st.image(stat_img, use_container_width=True)
                else:
                    st.warning("Stationarity plot not found. Re-run the pipeline.")

                # --- ACF / PACF plot ---
                st.markdown(f"#### ACF / PACF Analysis — {diag_compound}")
                acf_img = os.path.join(arima_diag_dir, f"acf_pacf_{diag_compound.lower()}.png")
                if os.path.exists(acf_img):
                    st.image(acf_img, use_container_width=True)
                else:
                    st.warning("ACF/PACF plot not found. Re-run the pipeline.")

                # --- Residual diagnostics ---
                st.markdown(f"#### Residual Diagnostics — {diag_compound}")
                resid_img = os.path.join(arima_diag_dir, f"residuals_{diag_compound.lower()}.png")
                if os.path.exists(resid_img):
                    st.image(resid_img, use_container_width=True)
                else:
                    st.warning("Residual plot not found. Re-run the pipeline.")

            # --- Cross-compound summary table ---
            st.markdown("---")
            st.markdown("#### All Compounds — Order Selection Summary")
            summary_rows = []
            for comp, cd in arima_diag.items():
                # After-differencing status: use combined verdict if available
                after_diff = cd.get("stationary_after_diff",
                                    cd.get("adf_differenced", {}).get("is_stationary", False))
                summary_rows.append({
                    "Compound": comp,
                    "Series Len": cd.get("series_length", "?"),
                    "Original": "❌ Non-stationary",
                    "d": cd.get("d_selected", "?"),
                    "After Diff": "✅ Stationary" if after_diff else "❌ Non-stationary",
                    "Order": f"ARIMA{tuple(cd.get('selected_order', []))}",
                    "AIC": round(cd.get("best_aic", 0), 2),
                })
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

        else:
            st.warning(
                "No ARIMA diagnostics found. Run the pipeline to generate them:"
            )
            st.code("python run_pipeline.py", language="bash")

    # ============================================================
    # Tab 5: About
    # ============================================================
    with tab5:
        st.markdown("""
        ### About This Project
        
        **F1 Pit Stop Strategy Predictor** is a time series analysis project that models 
        tire degradation in Formula 1 to predict optimal pit stop timing.
        
        #### Architecture
        
        ```
        FastF1 API -> Data Preprocessing -> Model Training -> Strategy Engine -> Frontend
        ```
        
        #### Data Source
        - **FastF1 API**: Official F1 timing data
        - **Seasons**: 2023-2024 (all races)
        - **Features**: Lap times, tire compounds, tire age, fuel effects, 
          sector times, track conditions, speed trap data
        
        #### Models (5 Total)
        
        | # | Model | Type | Eval Level | Description |
        |---|-------|------|------------|-------------|
        | 1 | **ARIMA** | Time Series | Curve | Autoregressive model on averaged degradation curves |
        | 2 | **VAR** | Multivariate TS | Curve | Vector Autoregression — models compounds as a system |
        | 3 | **XGBoost** | Machine Learning | Instance | Gradient boosted trees with feature interactions |
        | 4 | **LSTM** | Deep Learning | Instance | Bidirectional LSTM, autoregressive per-step |
        | 5 | **Hybrid** | Residual Stacking | Instance | ARIMA base + XGBoost residual correction |
        
        #### Key Features
        - **Fuel Correction**: Accounts for ~0.06s/lap fuel burn effect
        - **Degradation Modeling**: Quadratic + learned degradation curves
        - **Strategy Optimization**: Cost-benefit analysis for pit timing
        - **Multi-Compound**: Recommends compounds for each stint
        - **Regulation Compliance**: Ensures 2+ compound usage requirement
        
        #### How Pit Stop Decision Works
        
        1. Predict tire degradation curve for current compound
        2. When predicted degradation exceeds a **threshold** (configurable), 
           recommend pitting
        3. Choose next compound based on remaining race distance
        4. Repeat until race end
        
        #### Target: Australian GP 2025
        
        - **Circuit**: Albert Park, Melbourne
        - **Laps**: 58
        - **Characteristics**: Street-like circuit, medium abrasiveness
        
        ---
        
        *Built with FastF1, Streamlit, XGBoost, TensorFlow, and Statsmodels*
        """)


if __name__ == "__main__":
    main()
