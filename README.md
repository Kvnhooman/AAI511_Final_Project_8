# Music Composer Classification (Bach • Beethoven • Chopin • Mozart)

Classify classical **MIDI** excerpts by **composer** using two deep-learning approaches:

- **Model A:** LSTM (sequence model)  
- **Model B:** CNN→LSTM Hybrid (motif extractor + temporal model)

This repo uses a single “report-style” Jupyter notebook that includes rationale, code, outputs, and discussion inline.

---

## Live Demo

Try the web app here → **https://authenticcomposers.netlify.app**

Upload a `.mid` file and the app will analyze it to predict the most likely composer among **Bach, Beethoven, Chopin, Mozart** and show a confidence score.

> ⚠️ Supported input: MIDI (.mid). Other formats are not recognized by the current pipeline.

---

## Quick Start (Notebook)

### Google Colab (recommended)
1. Open the main notebook: `AAI511_Group8_Final_Assignment.ipynb`
2. Runtime → **GPU** (or TPU).
3. Run top-to-bottom. The notebook downloads/preps data (via Kaggle or uses pre-split pickles), trains, and evaluates both models.

### Local setup
```bash
# Clone
git clone https://github.com/Kvnhooman/AAI511_Final_Project_8.git
cd AAI511_Final_Project_8

# (Optional) create env
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install core deps (edit as needed)
pip install torch torchvision torchaudio scikit-learn numpy matplotlib pretty_midi music21 kagglehub
# Or pin exact versions from the notebook's Environment cell
```
Then open the notebook in Jupyter or VS Code and run all cells.

---

## Web App — How It Works (high level)

1. **Upload MIDI**: The app reads metadata and note events.  
2. **Preprocess**: Convert to model-ready tensors (pianoroll / token sequence), normalize length with windowing & padding.  
3. **Inference**: Run the trained model to produce class probabilities for the four composers.  
4. **Result**: Display the predicted composer and a confidence score. Optionally show top-4 ranking.

> Privacy note: Project code does **not** store uploads in this repository. If you fork/deploy your own backend, review your hosting provider’s data retention settings.

---

## Data

- **Source:** Kaggle dataset *MIDI classic music* (filtered to Bach, Beethoven, Chopin, Mozart).  
- **Format:** MIDI → converted to model-ready tensors (e.g., pianorolls / tokenized sequences).  
- **Splits:** Balanced **train/dev/test**; class balance is shown in the notebook.

> If you’re using pre-split pickle files (e.g., `lstm_dev.pkl`, `lstm_test.pkl`, etc.), place them in `data/` or update the paths near the data-loading cell.

---

## Results (test set, macro-averaged)

| Model  | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------:|:--------:|:---------------:|:------------:|:--------:|
| LSTM   | **0.7339** | **0.79** | **0.73** | **0.74** |
| Hybrid | 0.6894 | 0.76 | 0.69 | 0.69 |

**Notes**
- The ~**0.689** often referenced for the Hybrid corresponds to **macro-F1** (not overall accuracy).
- Hybrid shines on **Bach** (F1≈0.92) but under-recalls **Mozart** (~0.37). LSTM generalizes better overall.

---

## What’s Inside the Notebook

- **Objective & Rubric Map** – how the notebook satisfies the assignment criteria  
- **Dataset, Pre-processing, Feature Extraction** – MIDI → tensors, windowing, (optional) key-preserving transposition  
- **Models** – LSTM and CNN→LSTM hybrid, with hyperparameter notes  
- **Training** – loss/optimizer, dropout/weight decay, seeds, hardware notes  
- **Evaluation** – accuracy, macro-precision/recall/F1, confusion matrices  
- **Optimization & Error Analysis** – what we tuned; where models confuse classes and why  
- **Conclusion & Reproducibility** – summary, future work, environment versions  
- **References (APA)** – datasets, libraries, and foundational papers

---

## Repo Structure (typical)

```
.
├─ data/                                    # (optional) pre-split pickles or cached tensors
├─ AAI511_Group8_FinalAssignment.ipynb      # main report-style notebook (code + commentary)
├─ README.md                                # this file
└─ (optional) notebooks/                    # staging notebooks, experiments
```

---

## Reproducibility

- Use the **Environment** cell in the notebook to print Python/OS and library versions (PyTorch, scikit-learn, NumPy, Matplotlib, pretty_midi, music21).
- Set the runtime to **GPU** in Colab for faster training.
- Seeds are fixed in code where applicable.

---

## How to Cite

```
blanderbuss. (n.d.). MIDI classic music [Data set]. Kaggle. 
Retrieved August 11, 2025, from https://www.kaggle.com/datasets/blanderbuss/midi-classic-music
```

Additional references for LSTM, PyTorch, scikit-learn, NumPy, Matplotlib, and MIDI tooling are listed in the notebook’s **References** section (APA 7).

---

## Team

Kevin Hooman

Devin Eror
