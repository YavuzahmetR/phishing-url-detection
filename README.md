# PhiUSIIL Phishing URL Detection & Feature Analytics (V2)

This project is an advanced machine learning framework designed to detect phishing URLs using the **PhiUSIIL Phishing URL Dataset**. Unlike naive scraping models that easily cheat by analyzing raw HTML code from cached/dead pages, this system forces artificial intelligence to strictly evaluate **lightweight, network, and structural-level static URL elements**. This ensures ultra-fast, robust inference on live enterprise traffic—even when malicious endpoints are offline or guarded by anti-bot mechanics.

---

## Feature Selection Rationale & Data Integrity

To guarantee real-world generalization and avoid the **"Perfect Score Trap" (Data Leakage)**, the dataset's 56 original dimensions were aggressively pruned down to **14 pure structural features**:

### 1. Eliminated Features (Dropped for Security & Design)
* **`FILENAME`, `URL`, `Domain`, `TLD`, `Title`:** High-cardinality metadata strings dropped to prevent brute-force memory overfitting.
* **`URLSimilarityIndex`, `URLCharProb`, `TLDLegitimateProb`, `DomainTitleMatchScore`:** Pre-calculated proxy heuristics that leak the target label beforehand.
* **HTML/DOM-dependent counts (`LineOfCode`, `NoOfImage`, `NoOfJS`, `NoOfCSS`)**: Purged because real phishing sites use full-fledged graphical clones on active deployments, making raw DOM counts a scraping artifact.

### 2. Selected Structural Elements (The Final 14)
* **Network & Domain Identity:** `DomainLength`, `IsDomainIP`, `TLDLength`, `NoOfSubDomain`, `IsHTTPS`, `CharContinuationRate`.
* **Normalized Behavior Ratios:** `HasObfuscation`, `ObfuscationRatio`, `LetterRatioInURL`, `DegitRatioInURL`, `SpacialCharRatioInURL`.
* **Attack Query Tokens:** `NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL`.

---

## Target Variable & Inference Logic (Important)

**Label mapping:** Source `Label` = 0 means **Phishing**, 1 means **Legitimate**.  
Internally, we transform to `is_phishing` where `1` = Phishing, `0` = Legitimate.

This mapping is locked and tested to avoid any class-index confusion during inference.

---

## Preprocessing Strategy (Scaling)

Instead of a global `RobustScaler`, a `ColumnTransformer` is used during training. Only continuous numerical variables are scaled, while binary flags are passed through **untouched** using `passthrough`. This preserves the full discriminative power of every feature like `IsHTTPS`.

---

## Architectural Duel: Baselines vs. V2 Champion

To justify the engineering complexity of the V2 pipeline, a strict benchmarking duel was executed across three evolving architectural tiers on a locked validation partition:

### 1. Baseline Architectures 
* **Lexical Heuristic Baseline:** Evaluates traditional static triggers (e.g., presence of `@` symbols or raw IP addresses). While ultra-lightweight, it suffers from catastrophic **Recall deficiency (0.14)**, letting 86% of actual threats bypass security barriers.
* **LogReg Char N-gram Baseline:** A heavy character-level tokenization engine utilizing Logistic Regression. While highly accurate (0.99 F1), its massive memory footprint and token explosion make it structurally non-viable for sub-2ms microservice restrictions.

### 2. The V2 Champion (LightGBM + Platt Scaling)
Our final production model combines an optimized **LightGBM Classifier** with **Platt Scaling (Sigmoid Calibration)**. It utilizes only the 14 lightweight structural metrics, ensuring extreme inference speed while aggressively forcing a minimum **95% Recall safety target** on the phishing class to minimize high-risk False Negatives.

### 📊 Comprehensive Performance Matrix
The following real-time experimental matrix illustrates the model evolution on the evaluation set:

| Model Architecture | Overall Accuracy | Phishing Precision | Phishing Recall | F1-Score | Production Feasibility |
| :--- | :--- | :--- | :--- | :--- | :--- |
|  **Lexical Heuristic Baseline** | 0.68 | 0.83 | 0.14 | 0.24 | **High False Negative rate; misses 86% of actual threats.** |
|  **LogReg Char N-gram Baseline** | 1.00 | 1.00 | 0.99 | 0.99 | **Overfits to token text; prone to Zero-Day blindspots & high RAM overhead.** |
|  **LightGBM + Platt Scaling (V2)** | **0.98** | **1.00** | **0.96** | **0.98** | **Optimal balance of structural generalization and sub-2ms latency.)** |

### Hyperparameter Tuning via Optuna
The champion LightGBM engine underwent an automated Bayesian optimization loop:
* **Optimal Space Found:**
  ```json
  {"learning_rate": 0.0904, "num_leaves": 69, "max_depth": 8, "min_child_samples": 100, "subsample": 0.9810}
  ```
* **Final Threshold Selected:** `0.9514` (Calculated dynamically to lock high-confidence edge protection).

Detailed dataset bias analysis and architectural constraints are fully documented inside [reports/model_card.md](reports/model_card.md).

---

## Modular Directory Structure
```text
phishing-url-detection/
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions Automated CI Pipeline (Ruff + Pytest)
├── app/
│   └── main.py            # FastAPI Web Server (Accepts Raw URL Strings on-the-fly)
├── reports/
│   └── model_card.md      # Production Model Card & Dataset Bias Documentation
├── src/
│   └── phishing_guard/    # Core Encapsulated Framework
│       ├── data/          # Secure Data Splits (Domain-Grouped Split Walls)
│       ├── features/      # Real-time Lexical Feature Extraction Engine
│       └── modeling/      # Optuna Tuning, Platt Calibration & Artifact Managers
├── requirements.txt       # Frozen Environment Dependencies
├── run_experiment.py      # End-to-End Training, Tuning & Export Automation Script
└── README.md              # Project Documentation Manual
```

---

## Deployment & Execution Quickstart

### 1. Local Environment Provisioning
```bash
# Clone the repository architecture
git clone https://github.com
cd phishing-url-detection

# Install environment dependencies
pip install -r requirements.txt
```

### 2. Model Training, Calibration & Export
To execute the automated Optuna optimization loop, validate safety thresholds, and dump the calibrated single artifact model into `models/`, execute:
```bash
python run_experiment.py
```

### 3. Production Microservice Execution
To launch the ultra-fast FastAPI microservice natively via Uvicorn, execute:
```bash
uvicorn app.main:app --reload --port 8000
```

---

## Swagger Interactive UI
Once your server transitions to live execution status, navigate to: **http://localhost:8000/docs**

The V2 API layer accepts a raw, un-preprocessed URL string, automatically extracts structural tokens on-the-fly, and serves mathematical prediction outputs in under 2 milliseconds.

**Example Payload**
```json
{
  "url": "http://secure-bank-update-verify-checkpoint.com"
}
```

**Expected Response**
```json
{
  "is_phishing": true,
  "probability": 0.9999381527163119,
  "threshold": 0.9514251634442327,
  "model_version": "v2.0.0"
}
```
