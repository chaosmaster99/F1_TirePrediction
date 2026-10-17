# F1 Pit Stop Strategy Predictor

An F1 race strategy application that predicts tyre degradation and recommends pit stop strategies. The project includes a Streamlit interface, a FastF1 data pipeline, trained machine learning models, evaluation plots, and the accompanying paper/report artifacts.

## Features

- Interactive Streamlit dashboard for pit stop strategy recommendations
- FastF1 race and lap data collection
- Data preprocessing and feature engineering
- ARIMA, VAR, XGBoost, LSTM, and hybrid model implementations
- Model comparison and saved evaluation results
- Tyre compound and driver profile support

## Requirements

- Python 3.10 or newer
- Internet access when downloading new FastF1 session data
- Optional: a CUDA-enabled TensorFlow setup for faster LSTM training

Install the dependencies in a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the dashboard

The trained models and processed data are included in this repository, so the dashboard can be started directly:

```powershell
streamlit run app.py
```

Streamlit will display the local URL in the terminal, normally `http://localhost:8501`.

## Run the full pipeline

To fetch the configured 2023-2024 race data, preprocess it, train the models, compare their performance, and regenerate outputs:

```powershell
python run_pipeline.py
```

The pipeline stores raw and processed data in `data/`, trained models in `models/`, and plots and metrics in `plots/`. FastF1 session downloads are cached locally under `cache/`; this directory is intentionally excluded from Git because it can become several gigabytes in size.

## Run tests

```powershell
python -m pytest test_strategy.py
```

## Repository layout

```text
app.py                  Streamlit application
run_pipeline.py         End-to-end data and model pipeline
diag.py                 Diagnostics utility
src/                    Data, preprocessing, strategy, and model code
data/                   Raw and processed CSV data
models/                 Saved trained models and preprocessing artifacts
plots/                  Evaluation metrics and visualizations
paper/                  LaTeX paper and figures
ppt_report/             Generated report files
test_strategy.py        Strategy tests
requirements.txt        Python dependencies
```

## Notes

- The local virtual environment, FastF1 cache, Python bytecode, and pipeline logs are ignored by Git.
- Re-running the pipeline may replace the checked-in model and plot artifacts.
- The application is intended for research and strategy analysis, not live race control or betting decisions.