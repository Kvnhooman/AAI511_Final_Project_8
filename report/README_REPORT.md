# Report Runner

Use `Final_Report_Orchestrator.ipynb` as the single entry point.

## Configure
Edit **report_config.json** (or **params.py**) to set:
- RUN_PREPROCESS / RUN_HYBRID_TRAINING / RUN_LSTM_TRAINING
- KAGGLE_DATASET / ZIP_FILENAME
- OUTPUT_DIR (where processed splits are written)
- TARGET_COMPOSERS

## Run
Open the orchestrator in Colab and Run All. The notebook will:
1) Preprocess via KaggleHub and create standardized splits under OUTPUT_DIR.
2) (Optionally) train Hybrid and/or LSTM models.
3) Run the evaluation notebook.
4) Display a models table and confusion matrices.

Artifacts:
- `/mnt/data/artifacts/metrics_*.json`
- `/mnt/data/artifacts/confusion_*.png`
- Splits in OUTPUT_DIR: `lstm_data.pkl`, `lstm_dev.pkl`, `lstm_test.pkl`, `label_encoder.pkl`
