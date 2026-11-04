# Malware Detection and Malicious URL Classification with Deep Learning and XAI

Code repository developed as part of a Master's dissertation in Informatics (MINF) at Universidade da Maia (UMAIA), supervised by Professors Sérgio Silva.

This project explores the application of Deep Learning and Explainable AI (XAI) techniques to two cybersecurity problems:

1. **Malware classification** based on PE (Portable Executable) file metadata and runtime behavior, at three levels of granularity (Class, Category, and Family).
2. **Malicious URL detection**, using a hybrid model that combines a tabular neural network with a transformer-based model (DistilBERT).

In both cases, SHAP (*SHapley Additive exPlanations*) is used to generate interpretable explanations for the models' decisions.

---

## Repository structure

```
.
├── Class:Category:Family/
│   ├── Category/
│   │   ├── Category.py           # Multiclass classification: Benign / Ransomware / RAT / Stealer / Trojan
│   │   └── ransomwareNovo.xlsx.zip
│   ├── Class/
│   │   ├── Class.py              # Binary classification: Benign vs Malware
│   │   └── ransomware.xlsx.zip
│   └── Family/
│       ├── Family.py             # Multiclass classification: 26 malware families
│       └── ransomwareNovo.xlsx.zip
├── CNN malicious_urls/
│   ├── main3CNN.py               # Malicious URL detection (hybrid tabular + DistilBERT model)
│   └── malicious_urls.csv.zip
├── .gitignore
└── README.md
```

> Datasets are provided as `.zip` archives to keep the repository size manageable. Unzip each `.xlsx.zip` / `.csv.zip` file in place before running the corresponding script (see [How to run](#how-to-run) below).

---

## Datasets

| Dataset | File | Samples | Description |
|---|---|---|---|
| Malware (PE + behavioral) | `ransomware.xlsx` | 21,752 (10,876 benign / 10,876 malicious) | 67 features: 49 static (PE Header) + 18 behavioral |
| Malware (malicious subset) | `ransomwareNovo.xlsx` | 10,876 malicious samples | Same features, used to predict `Category` (4 classes) and `Family` (26 classes) |
| URLs | `malicious_urls.csv` | 651,191 | Columns `url` and `type` (benign, defacement, phishing, malware); reduced to binary classification (benign vs malicious) |

> Note: the datasets are included in this repository solely for the reproducibility of the dissertation results.

---

## Scripts

### `Class.py` — Benign vs Malware (binary)
Fully connected neural network (MLP) with `BCELoss`, `StandardScaler` normalization, early stopping, and decision threshold calibration on the validation set (macro F1 maximization). Includes per-sample SHAP explanations.

### `Category.py` — Malware category (multiclass, 4 classes)
MLP with `CrossEntropyLoss`, `BatchNorm`, and `Dropout`, early stopping, and evaluation using macro AUC (OvR), macro F1, and macro Recall. Classes: `Ransomware`, `RAT`, `Stealer`, `Trojan`.

### `Family.py` — Malware family (multiclass, 26 classes)
Similar architecture to `Category.py`, with greater capacity (256 → 128 → 64), suited to the larger number of classes.

### `main3CNN.py` — Malicious URL detection (hybrid model)
Combines:
- A tabular neural network trained on features extracted from the URL (length, entropy, number of dots/hyphens, presence of an IP address, etc.);
- A DistilBERT model fine-tuned on the URL text.

The final risk score results from a weighted fusion of both models (63% DistilBERT + 37% tabular model), classified as `BLOCK` / `WARN` / `ALLOW` depending on the risk score.

---

## Requirements

- Python 3.10+
- Use of a virtual environment is recommended

Install dependencies:

```bash
pip install pandas numpy scikit-learn torch transformers shap matplotlib tldextract openpyxl
```

> If a CUDA-enabled GPU is available, `torch` will automatically use it (`device = "cuda" if torch.cuda.is_available() else "cpu"`).

---

## How to run

Each script is self-contained and should be run from the folder containing its corresponding dataset. Unzip the dataset first, then run the script from inside its folder:

```bash
# Malware detection (Benign vs Malware)
cd "Class:Category:Family/Class"
unzip ransomware.xlsx.zip
python Class.py

# Malware category classification
cd "Class:Category:Family/Category"
unzip ransomwareNovo.xlsx.zip
python Category.py

# Malware family classification
cd "Class:Category:Family/Family"
unzip ransomwareNovo.xlsx.zip
python Family.py

# Malicious URL detection
cd "CNN malicious_urls"
unzip malicious_urls.csv.zip
python main3CNN.py
```

Each run will:
1. Load and preprocess the corresponding dataset;
2. Train the model (with early stopping in the `Class.py`, `Category.py`, and `Family.py` scripts);
3. Generate a loss curve plot (`loss_curve_*.png`, automatically saved by `Class.py`, `Category.py`, and `Family.py`);
4. Print evaluation metrics on the test set (`classification_report`, AUC, F1, Recall);
5. Generate SHAP explanations for a set of samples, highlighting the main features driving each classification.

> **Note on runtime:** the scripts are configured with `EPOCHS = 1000` and a high `PATIENCE` value, so training may take a considerable amount of time, especially without a GPU. Adjust these values as needed for quick tests.

---

## Explainability (XAI)

All models use SHAP to generate sample-level explanations. Feature contributions are translated into human-readable phrases (`FEATURE_TEXT` dictionary in each script), making the results easier to interpret for non-technical users.

---

## Related publication

Part of this work was presented at the ICCET 2026 conference, in the paper *"Explainable Hybrid Transformer Framework for Malicious URL Detection"*.

---

## Author

Alberto João Ribeiro Mendonça
Master's in Informatics — Universidade da Maia (UMAIA)
Supervisors: Prof. Sérgio Silva
