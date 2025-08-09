import io
import os
import math
import tempfile
from typing import Dict, Tuple, List

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Optional imports for real model inference
ORT_AVAILABLE = False
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ORT_AVAILABLE = False

PM_AVAILABLE = True
try:
    import pretty_midi
except Exception:
    PM_AVAILABLE = False

LABELS = ["Bach", "Beethoven", "Chopin", "Mozart"]

app = Flask(__name__, static_folder="../web", static_url_path="")
CORS(app)

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def _time_weighted_polyphony(notes: List[Tuple[float, float]]) -> float:
    # notes: list of (start, end)
    if not notes:
        return 0.0
    events = []
    for s, e in notes:
        events.append((s, 1))
        events.append((e, -1))
    events.sort()
    active = 0
    prev_t = events[0][0]
    total = 0.0
    for t, delta in events:
        dt = max(0.0, t - prev_t)
        total += active * dt
        active += delta
        prev_t = t
    duration = max(1e-6, events[-1][0] - events[0][0])
    return total / duration

def extract_features(midi_bytes: bytes) -> Dict[str, float]:
    if not PM_AVAILABLE:
        raise RuntimeError("pretty_midi is not installed. Please install dependencies.")
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tmp:
        tmp.write(midi_bytes)
        tmp.flush()
        pm = pretty_midi.PrettyMIDI(tmp.name)

    # Collect notes
    notes = []
    velocities = []
    pitch_classes = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append((n.start, n.end, n.pitch, n.velocity))
            velocities.append(n.velocity)
            pitch_classes.append(n.pitch % 12)

    duration = pm.get_end_time() if pm.get_end_time() > 0 else 1e-6
    n_notes = len(notes)
    note_density = n_notes / duration

    # Tempo statistics
    try:
        t_changes, tempi = pm.get_tempo_changes()
        if len(tempi) == 0:
            tempi = np.array([pm.estimate_tempo()])
        tempo_mean = float(np.mean(tempi))
        tempo_std = float(np.std(tempi))
    except Exception:
        tempo_mean = float(pm.estimate_tempo()) if hasattr(pm, "estimate_tempo") else 120.0
        tempo_std = 0.0

    # Velocity stats
    if len(velocities) == 0:
        vel_mean, vel_std = 0.0, 0.0
    else:
        vel_mean = float(np.mean(velocities))
        vel_std = float(np.std(velocities))

    # Pitch class histogram (12)
    if len(pitch_classes) == 0:
        pch = np.zeros(12, dtype=float)
    else:
        pch = np.bincount(np.array(pitch_classes), minlength=12).astype(float)
        if pch.sum() > 0:
            pch /= pch.sum()

    # Polyphony (time-weighted avg number of simultaneous notes)
    polyphony = _time_weighted_polyphony([(s, e) for (s, e, _, _) in notes])

    feats = {
        "duration_s": float(duration),
        "notes": int(n_notes),
        "note_density": float(note_density),
        "tempo_mean": float(tempo_mean),
        "tempo_std": float(tempo_std),
        "velocity_mean": float(vel_mean),
        "velocity_std": float(vel_std),
        "polyphony": float(polyphony),
    }
    for i in range(12):
        feats[f"pc_{i}"] = float(pch[i])
    return feats

class DemoHeuristicModel:
    """
    Lightweight fallback when no real model is provided.
    Produces a stable, deterministic score from features.
    """
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        # x expected shape (D,)
        # Simple crafted logits for showcase
        # Emphasize baroque polyphony for Bach, dynamic variance for Beethoven,
        # rubato/tempo variance & chromatic use for Chopin, diatonic simplicity for Mozart.
        D = x.shape[0]
        # Index mapping (keep in sync with extract_features)
        idx = {
            "note_density": 2,
            "tempo_mean": 3,
            "tempo_std": 4,
            "velocity_std": 6,
            "polyphony": 7,
            # pitch classes start at 8 -> 19
            "pc_start": 8
        }
        pc = x[idx["pc_start"]:idx["pc_start"]+12]
        # Diatonic emphasis: C, D, E, F, G, A, B -> 0,2,4,5,7,9,11
        diatonic = pc[[0,2,4,5,7,9,11]].sum()
        chromatic = pc.sum() - diatonic

        logits = np.zeros(4, dtype=float)
        # Bach
        logits[0] = 2.2*x[idx["polyphony"]] + 1.5*x[idx["note_density"]] + 0.3*diatonic - 0.2*x[idx["tempo_std"]]
        # Beethoven
        logits[1] = 1.8*x[idx["velocity_std"]] + 1.7*x[idx["tempo_std"]] + 0.5*x[idx["note_density"]]
        # Chopin
        logits[2] = 1.9*chromatic + 1.2*x[idx["tempo_std"]] + 0.4*(x[idx["velocity_std"]])
        # Mozart
        logits[3] = 2.0*diatonic + 0.8*(1.0 - x[idx["polyphony"]]) + 0.2*(1.0 - chromatic)

        # Normalize and softmax
        probs = _softmax(logits)
        return probs

class InferenceEngine:
    def __init__(self):
        self.mode = "DEMO"
        self.labels = LABELS
        self.ort_sess = None

        model_path = os.environ.get("ONNX_MODEL_PATH", os.path.join(os.path.dirname(__file__), "../models/composer_classifier.onnx"))
        if ORT_AVAILABLE and os.path.isfile(model_path):
            try:
                self.ort_sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                self.mode = "ONNX"
                print(f"[INFO] Loaded ONNX model from {model_path}")
            except Exception as e:
                print(f"[WARN] Failed to load ONNX model ({e}). Falling back to DEMO.")

        if self.mode == "DEMO":
            print("[INFO] Running in DEMO mode (heuristic). Place an ONNX model at ../models/composer_classifier.onnx to enable real inference.")
            self.demo = DemoHeuristicModel()

    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        # Arrange feature vector in a fixed order
        feature_keys = [
            "duration_s","notes","note_density","tempo_mean","tempo_std","velocity_mean","velocity_std","polyphony",
            *[f"pc_{i}" for i in range(12)]
        ]
        x = np.array([features[k] for k in feature_keys], dtype=np.float32)

        if self.mode == "ONNX" and self.ort_sess is not None:
            try:
                # Assumes a single input and single output with probabilities or logits
                input_name = self.ort_sess.get_inputs()[0].name
                out = self.ort_sess.run(None, {input_name: x.reshape(1, -1)})[0].squeeze()
                # If output looks like logits, softmax them
                if out.ndim == 1:
                    if not np.all((0.0 <= out) & (out <= 1.0)) or abs(out.sum() - 1.0) > 1e-3:
                        out = _softmax(out)
                    probs = out
                else:
                    probs = out[0]
            except Exception as e:
                print(f"[WARN] ONNX inference failed: {e}. Falling back to DEMO.")
                probs = self.demo.predict_proba(x)
        else:
            probs = self.demo.predict_proba(x)

        probs = probs.astype(float)
        probs = probs / (probs.sum() + 1e-9)
        result = {label: float(p) for label, p in zip(self.labels, probs)}
        return result

ENGINE = InferenceEngine()

@app.route("/")
def root():
    # Serve the static index.html
    return app.send_static_file("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    if "midi" not in request.files:
        return jsonify({"error": "No file uploaded under form field 'midi'."}), 400
    f = request.files["midi"]
    if f.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    data = f.read()
    if len(data) == 0:
        return jsonify({"error": "Empty file."}), 400

    try:
        feats = extract_features(data)
        probs = ENGINE.predict(feats)
        # Authenticity heuristic based on confidence
        best_label = max(probs, key=probs.get)
        confidence = float(probs[best_label])
        if confidence >= 0.75:
            authenticity = "Likely authentic"
        elif confidence >= 0.55:
            authenticity = "Possibly authentic"
        else:
            authenticity = "Uncertain / possibly misattributed"

        return jsonify({
            "composer": best_label,
            "confidence": confidence,
            "probabilities": probs,
            "authenticity": authenticity,
            "features": feats,
            "engine": ENGINE.mode
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process MIDI: {e}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)