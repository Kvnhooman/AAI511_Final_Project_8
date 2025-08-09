# Composer Classifier Demo (LSTM+CNN)

A tiny Flask app + single-page UI to showcase composer prediction for MIDI files.
Supports Bach / Beethoven / Chopin / Mozart.

## Quick Start

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in your browser and upload a MIDI file.

### Using your real model
- Export your trained model to ONNX (single input feature vector).
- Place it at: `models/composer_classifier.onnx` (relative to this project root).
- Restart the server. The status pill will show `Engine: ONNX` when loaded.
- Update `extract_features()` if your model expects a different feature set.

### Notes
- If no ONNX model is found, the server runs a deterministic **heuristic demo** for probabilities.
- Edit thresholds or labels in `backend/app.py` to match your training.
