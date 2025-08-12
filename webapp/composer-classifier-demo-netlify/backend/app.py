import io, os, json, math, tempfile
from typing import Dict, Tuple, List

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------- Runtime deps ----------
import pretty_midi

ORT_AVAILABLE = True
try:
    import onnxruntime as ort
except Exception:
    ORT_AVAILABLE = False

# ---------- Config ----------
LABELS_FALLBACK = ["bach", "beethoven", "chopin", "mozart"]  # lowercase for consistency
WINDOW_LEN = 50
STRIDE = 1
PITCH_START, PITCH_END = 21, 108  # inclusive (88 keys)
NUM_PITCHES = PITCH_END - PITCH_START + 1
BATCH = 64

# ---------- Flask ----------
app = Flask(__name__, static_folder="../web", static_url_path="")
CORS(app)

# ---------- Utilities ----------
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def load_labels(model_dir: str) -> List[str]:
    lp = os.path.join(model_dir, "labels.json")
    if os.path.isfile(lp):
        try:
            with open(lp, "r") as f:
                labs = json.load(f)
            # normalize capitalization for UI
            return [s.capitalize() for s in labs]
        except Exception:
            pass
    return [s.capitalize() for s in LABELS_FALLBACK]

def midi_to_events(midi_bytes: bytes) -> np.ndarray:
    """
    Return (N,3) float32: [pitch, step, duration], step = time since previous onset.
    """
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tmp:
        tmp.write(midi_bytes)
        tmp.flush()
        pm = pretty_midi.PrettyMIDI(tmp.name)

    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append((n.start, n.end, n.pitch))
    if not notes:
        return None
    notes.sort(key=lambda x: x[0])

    ev = []
    prev_onset = notes[0][0]
    for start, end, pitch in notes:
        step = max(0.0, start - prev_onset)
        dur = max(0.0, end - start)
        ev.append([float(pitch), float(step), float(dur)])
        prev_onset = start
    return np.asarray(ev, dtype=np.float32)

def windows_from_events(ev: np.ndarray, win=WINDOW_LEN, stride=STRIDE) -> np.ndarray:
    """
    Slice (N,3) into windows -> (num_windows, win, 3)
    """
    if ev is None or len(ev) < win:
        return np.empty((0, win, 3), dtype=np.float32)
    n = 1 + (len(ev) - win) // stride
    out = np.empty((n, win, 3), dtype=np.float32)
    j = 0
    for i in range(0, len(ev) - win + 1, stride):
        out[j] = ev[i:i+win]
        j += 1
    return out

def seq_to_pianoroll_fast(seq: np.ndarray) -> np.ndarray:
    """
    seq: (T,3) where seq[:,0]=pitch -> return (1,88,T)
    """
    T = seq.shape[0]
    pr = np.zeros((1, NUM_PITCHES, T), dtype=np.float32)
    pitches = seq[:, 0].astype(np.int16)
    idx = pitches - PITCH_START
    t_idx = np.arange(T, dtype=np.int32)
    mask = (idx >= 0) & (idx < NUM_PITCHES)
    pr[0, idx[mask], t_idx[mask]] = 1.0
    return pr

# ---------- Inference Engine ----------
class Engine:
    def __init__(self):
        self.mode = "DEMO"
        self.labels = [s.capitalize() for s in LABELS_FALLBACK]
        self.ort_sess = None
        self.is_hybrid = False  # False = LSTM-only, True = Hybrid

        model_path = os.environ.get(
            "ONNX_MODEL_PATH",
            os.path.join(os.path.dirname(__file__), "../models/composer_classifier.onnx"),
        )
        model_dir = os.path.dirname(model_path)

        if ORT_AVAILABLE and os.path.isfile(model_path):
            try:
                self.ort_sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                # detect model type from inputs
                inps = [i.name for i in self.ort_sess.get_inputs()]
                self.is_hybrid = (len(inps) >= 2)  # expect ["x_lstm","x_cnn"] for hybrid
                self.mode = "ONNX"
                self.labels = load_labels(model_dir)
                print(f"[INFO] ONNX loaded: {model_path} | Hybrid={self.is_hybrid} | Labels={self.labels}")
            except Exception as e:
                print(f"[WARN] ONNX load failed: {e}; falling back to DEMO")

        if self.mode == "DEMO":
            print("[INFO] DEMO engine active (no ONNX).")

    def predict_probs(self, midi_bytes: bytes) -> Dict[str, float]:
        # Build windows
        ev = midi_to_events(midi_bytes)
        X_l = windows_from_events(ev, WINDOW_LEN, STRIDE)  # (Nw, 50, 3)
        if X_l.shape[0] == 0:
            raise ValueError("Not enough notes to form a 50-note window.")

        if self.mode == "ONNX" and self.ort_sess is not None:
            # Run in batches, average logits
            logits_all = []
            if self.is_hybrid:
                # Build matching pianorolls
                X_c = np.stack([seq_to_pianoroll_fast(seq) for seq in X_l], axis=0)  # (Nw,1,88,50)
                for i in range(0, X_l.shape[0], BATCH):
                    xb_l = X_l[i:i+BATCH].astype(np.float32)          # (B,50,3)
                    xb_c = X_c[i:i+BATCH].astype(np.float32)          # (B,1,88,50)
                    feeds = {}
                    # be flexible with input names
                    inp0 = self.ort_sess.get_inputs()[0].name
                    feeds[inp0] = xb_l
                    if len(self.ort_sess.get_inputs()) >= 2:
                        inp1 = self.ort_sess.get_inputs()[1].name
                        feeds[inp1] = xb_c
                    out = self.ort_sess.run(None, feeds)[0]           # (B,C) logits or probs
                    logits_all.append(out.astype(np.float32))
            else:
                for i in range(0, X_l.shape[0], BATCH):
                    xb_l = X_l[i:i+BATCH].astype(np.float32)          # (B,50,3)
                    feeds = { self.ort_sess.get_inputs()[0].name: xb_l }
                    out = self.ort_sess.run(None, feeds)[0]           # (B,C)
                    logits_all.append(out.astype(np.float32))

            logits = np.concatenate(logits_all, axis=0)               # (Nw,C)
            # If they are already probs (0-1 and ~sum=1), keep; else softmax
            row_sums = logits.sum(axis=1, keepdims=True)
            if np.any((logits < 0) | (logits > 1)) or np.any(np.abs(row_sums - 1) > 1e-3):
                probs_w = softmax(logits)                             # (Nw,C)
            else:
                probs_w = logits
            probs = probs_w.mean(axis=0)                               # (C,)
        else:
            # DEMO fallback: simple hand-crafted signal from windows
            # Use polyphony proxy via pianoroll occupancy & step variability
            def demo_scores(seq: np.ndarray) -> np.ndarray:
                pr = seq_to_pianoroll_fast(seq)                       # (1,88,50)
                poly = (pr.sum(axis=1) > 1e-6).mean()                 # fraction of frames with any notes
                step_std = np.std(seq[:, 1]) if seq.shape[0] else 0.0
                dur_std  = np.std(seq[:, 2]) if seq.shape[0] else 0.0
                logits = np.array([
                    1.8*poly + 0.2*(1-step_std),     # Bach-ish polyphony
                    1.5*dur_std + 1.2*step_std,      # Beethoven dynamics-ish
                    1.4*step_std + 0.6*dur_std,      # Chopin rubato-ish
                    1.7*(1-poly) + 0.3*(1-step_std), # Mozart simpler textures
                ], dtype=np.float32)
                return logits
            L = np.stack([demo_scores(s) for s in X_l], axis=0)       # (Nw,4)
            probs = softmax(L).mean(axis=0)

        # Normalize and map to labels
        probs = probs.astype(np.float64)
        probs = probs / (probs.sum() + 1e-9)
        return {label: float(p) for label, p in zip(self.labels, probs)}

ENGINE = Engine()

# ---------- Routes ----------
@app.route("/")
def root():
    return app.send_static_file("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    if "midi" not in request.files:
        return jsonify({"error": "No file uploaded under form field 'midi'."}), 400
    f = request.files["midi"]
    if f.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    data = f.read()
    if not data:
        return jsonify({"error": "Empty file."}), 400
    if len(data) > 10 * 1024 * 1024:
        return jsonify({"error": "File too large (>10MB)."}), 413

    try:
        probs = ENGINE.predict_probs(data)
        best = max(probs, key=probs.get)
        conf = float(probs[best])
        if conf >= 0.75:
            authenticity = "Likely authentic"
        elif conf >= 0.55:
            authenticity = "Possibly authentic"
        else:
            authenticity = "Uncertain / possibly misattributed"

        return jsonify({
            "composer": best,
            "confidence": conf,
            "probabilities": probs,
            "authenticity": authenticity,
            "engine": ENGINE.mode,
            "hybrid": ENGINE.is_hybrid
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process MIDI: {e}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)