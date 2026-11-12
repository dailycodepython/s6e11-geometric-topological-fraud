# s6e11-geometric-topological-fraud
Math over brute compute: Optimal Transport, Persistent Homology (TDA), Geodesic Manifolds &amp; SHAP for Kaggle S6E11 Credit Card Fraud Detection


**Philosophy:** *Math over brute compute.*  
A lean, CPU-friendly machine learning pipeline for **Kaggle Playground Series S6E11 (Credit Card Fraud Detection)**, combining Optimal Transport, Topological Data Analysis (TDA), Geodesic Manifolds, and Cooperative Game Theory (SHAP) on top of LightGBM.

> **Key Empirical Result:** Adding 10 geometric features (+6 Optimal Transport, +4 Graph Dijkstra) expanded the feature space from 24 to 34 columns and boosted validation **ROC AUC from 0.94396 to 0.96599 (+0.02203)** on 699,960 transactions.


---

## 1. Problem & Data Architecture
* **Dataset:** ~1,000,000 synthetic financial transactions (`699,960` train / `299,984` test) across 24 tabular features.
* **Class Imbalance:** Extreme asymmetry with **1.67% fraud** (`11,689` positive cases vs. `688,271` normal transactions).
* **Evaluation Metric:** ROC AUC (supported by Balanced Accuracy diagnostics for threshold calibration).

---

## 2. Mathematical Pipeline

### I. Bayesian Prior Calibration
To counter the severe 1.67% class prior without synthetic oversampling, predicted posterior probabilities $P(c \mid x)$ are calibrated against empirical class priors $P(c)$:
$$P_{\text{calibrated}}(c \mid x) \propto \frac{P(c \mid x)}{P(c)}$$
* **Impact:** Boosted local validation Balanced Accuracy from **0.65268** to **0.86874** (+21.6%) out of the box.

### II. Optimal Transport & Sinkhorn Potentials (`POT`)
* **Covariate Shift Audit:** Computed exact squared Wasserstein distance ($W_2^2$) between standardized 13-dimensional numerical spaces of `train` and `test` via Earth Mover's Distance (`ot.emd2`). Result: **$W_2^2 = 3.670105$**, confirming geometric stability across splits.
* **Transport Features:** Vectorized batch computation of squared transport cost and entropic Sinkhorn soft-min potentials relative to reference `Normal` and `Fraud` point clouds.

### III. Geodesic Graph Distances & Non-Linear Manifolds (`SciPy` & `Isomap`)
* **k-NN Dijkstra Geodesics:** Constructed a sparse 10-nearest-neighbor graph across the combined manifold (~1M vertices) and computed batched shortest geodesic paths to class anchor nodes.
* **Isomap Projection:** Fitted a stratified 3-component Isomap embedding (preserving rare fraud cluster topology) to project transactions into intrinsic geodesic coordinates.

### IV. Topological Data Analysis (`Ripser`)
* Extracted persistent homology features ($H_0$ connected components and $H_1$ 1-dimensional loops/cycles) via Vietoris–Rips filtration over overlapping sliding windows, summarizing topological lifetimes and **Persistent Entropy**.

### V. Cooperative Game Theory Audit (`SHAP`)
* Evaluated marginal feature contributions across coalitions using `TreeExplainer` on a balanced validation subset to verify orthogonal signal gain and eliminate redundant features.

---

## 3. Validation Results (LightGBM)

| Experiment Stage | Validation ROC AUC | Balanced Accuracy | Notes |
| :--- | :---: | :---: | :--- |
| **1. Baseline LightGBM** (24 raw features) | `0.94396` | `0.65268` | Single split (`random_state=42`) |
| **2. Baseline + Bayesian Calibration** | `0.94396` | `0.86874` | Prior probability shift |
| **3. Extended Geometry** (+6 OT, +4 Dijkstra) | **0.96599** | — | **+0.02203 AUC** gain (34 features total) |
| **4. Full 5-Fold Ensemble** (+ Ripser TDA) | **`[TBD]`** | — | Final OOF score & Kaggle submission |

---

## 4. Tech Stack & Lean Constraints
* **Environment:** Google Colab (CPU-only, `float32` memory optimization, batched distance matrices).
* **Libraries:** `lightgbm`, `POT` (`ot`), `ripser`, `persim`, `scikit-learn`, `scipy`, `shap`.
