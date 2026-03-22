"""
Quantum-Powered Intrusion Detection System (Q-IDS)
===================================================
Dataset  : UNSW-NB15
Approach : Quantum Kernel SVM (ZZFeatureMap / Angle Encoding)
Hardware : Classical simulation with 2 or 4 qubits (PennyLane default.qubit)

Architecture
------------
1. Data Loading & EDA
2. Preprocessing
   - Drop ID/label-string columns
   - Label-encode categoricals
   - Feature selection (variance + mutual-info top-N)
   - PCA down to N_QUBITS dimensions  ← maps to qubit count
   - Min-Max scale to [0, π]
3. Quantum Kernel computation
   - ZZFeatureMap-style circuit with entanglement (4 qubits)
   - OR simple AngleEmbedding kernel (2 qubits) -- set N_QUBITS=2
   - Kernel matrix K[i,j] = |<ψ(xᵢ)|ψ(xⱼ)>|²
4. SVM with precomputed quantum kernel
5. Evaluation: Accuracy, F1, Precision, Recall, AUC-ROC, Confusion Matrix

Key fixes vs. the original notebook
--------------------------------------
- Quantum weights ARE used (kernel trick -- no random uninitialised weights)
- True quantum kernel, not a random projection
- Proper evaluation metrics (not just accuracy on imbalanced data)
- Stratified split from the full dataset
- Reproducible (all random seeds fixed)
- Modular, documented code

Usage
-----
  pip install pennylane scikit-learn pandas numpy matplotlib seaborn

  # Place dataset CSVs in the same folder, or update DATA_DIR below.
  # Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset
  #   UNSW_NB15_training-set.csv
  #   UNSW_NB15_testing-set.csv

  python quantum_ids.py              # runs with 4 qubits (default)
  python quantum_ids.py --qubits 2   # runs with 2 qubits (faster)
"""

import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend -- safe on any machine
import matplotlib.pyplot as plt
import seaborn as sns

import pennylane as qml

from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC

# ─────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────
RANDOM_SEED   = 42
TRAIN_CSV     = "UNSW_NB15_training-set.csv"
TEST_CSV      = "UNSW_NB15_testing-set.csv"
TRAIN_SAMPLES = 600   # quantum kernel is O(n²) -- keep tractable
TEST_SAMPLES  = 200
TOP_FEATURES  = 20    # mutual-info pre-filter before PCA

np.random.seed(RANDOM_SEED)

# Categorical columns that need label encoding
CATEGORICAL_COLS = ["proto", "service", "state"]

# Columns to drop (IDs, attack category string, etc.)
DROP_COLS = ["id", "attack_cat"]   # 'label' is the target


# ─────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────
def load_data(train_csv: str, test_csv: str) -> pd.DataFrame:
    """Load and combine train/test CSVs."""
    print("► Loading data...")
    train = pd.read_csv(train_csv)
    test  = pd.read_csv(test_csv)
    df    = pd.concat([train, test], ignore_index=True)
    print(f"  Combined shape : {df.shape}")
    print(f"  Class balance  : {df['label'].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────
# 2.  PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame, n_qubits: int, top_features: int = TOP_FEATURES):
    """
    Full preprocessing pipeline.
    Returns scaled PCA-reduced arrays ready for quantum encoding.
    """
    print("\n► Preprocessing...")

    # 2a. Drop useless columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # 2b. Encode categoricals
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # 2c. Separate features / target
    y = df["label"].values
    X = df.drop(columns=["label"]).select_dtypes(include=[np.number])

    # 2d. Fill NaNs with column median
    X = X.fillna(X.median())

    print(f"  Numeric features available : {X.shape[1]}")

    # 2e. Mutual-information feature selection
    mi = mutual_info_classif(X, y, random_state=RANDOM_SEED)
    top_idx = np.argsort(mi)[::-1][:top_features]
    X = X.iloc[:, top_idx]
    print(f"  After MI selection         : {X.shape[1]} features")

    # 2f. Stratified sample (keep quantum kernel tractable)
    n_total = TRAIN_SAMPLES + TEST_SAMPLES
    X_s, _, y_s, _ = train_test_split(
        X, y, train_size=n_total, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y_s,
        test_size=TEST_SAMPLES / n_total,
        stratify=y_s,
        random_state=RANDOM_SEED
    )
    print(f"  Train : {len(X_train)}  |  Test : {len(X_test)}")
    print(f"  Train class balance : {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Test  class balance : {dict(zip(*np.unique(y_test,  return_counts=True)))}")

    # 2g. PCA to exactly n_qubits dimensions
    pca = PCA(n_components=n_qubits, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)
    explained   = pca.explained_variance_ratio_.sum()
    print(f"  PCA to {n_qubits} dims — variance explained : {explained:.1%}")

    # 2h. Scale to [0, π]  (required for angle encoding)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled  = scaler.transform(X_test_pca)

    return X_train_scaled, X_test_scaled, y_train, y_test


# ─────────────────────────────────────────────
# 3.  QUANTUM KERNEL
# ─────────────────────────────────────────────
def build_kernel_circuit(n_qubits: int):
    """
    Builds a quantum kernel circuit.

    For n_qubits == 2  →  AngleEmbedding (RY) + CNOT entanglement
    For n_qubits == 4  →  ZZFeatureMap-style double-encoding with full entanglement

    The kernel value K(x, y) = |<0|U†(y) U(x)|0>|²
    is computed via the 'overlap' (swap-test equivalent via adjoint).
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    def feature_map(x):
        """Encode a single data point x into quantum state."""
        if n_qubits == 2:
            # ── 2-qubit: simple AngleEmbedding + CNOT ──────────────
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            qml.CNOT(wires=[0, 1])
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Z")
        else:
            # ── 4-qubit: ZZFeatureMap (2 layers) ───────────────────
            # Layer 1 – Hadamard + single-qubit rotation
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(2.0 * x[i], wires=i)

            # ZZ interactions (all pairs)
            pairs = [(i, j) for i in range(n_qubits) for j in range(i+1, n_qubits)]
            for i, j in pairs:
                qml.CNOT(wires=[i, j])
                qml.RZ(2.0 * (np.pi - x[i]) * (np.pi - x[j]), wires=j)
                qml.CNOT(wires=[i, j])

            # Layer 2 (repeat for expressibility)
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(2.0 * x[i], wires=i)

            for i, j in pairs:
                qml.CNOT(wires=[i, j])
                qml.RZ(2.0 * (np.pi - x[i]) * (np.pi - x[j]), wires=j)
                qml.CNOT(wires=[i, j])

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        """Compute overlap <ψ(x2)|ψ(x1)> via adjoint trick."""
        feature_map(x1)
        qml.adjoint(feature_map)(x2)
        return qml.probs(wires=range(n_qubits))

    def kernel(x1, x2):
        """Return |<ψ(x1)|ψ(x2)>|² = probability of measuring |0...0>."""
        probs = kernel_circuit(x1, x2)
        return float(probs[0])   # |0...0> state probability

    return kernel


def compute_kernel_matrix(X1, X2, kernel_fn, label=""):
    """Compute the full kernel matrix K[i,j] = k(X1[i], X2[j])."""
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    total = n1 * n2
    start = time.time()
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_fn(X1[i], X2[j])
        elapsed = time.time() - start
        eta = elapsed / (i + 1) * (n1 - i - 1)
        print(f"\r  {label} kernel matrix: row {i+1}/{n1}  |  "
              f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s   ", end="", flush=True)
    print()
    return K


# ─────────────────────────────────────────────
# 4.  TRAINING & EVALUATION
# ─────────────────────────────────────────────
def train_and_evaluate(K_train, K_test, y_train, y_test):
    """Train SVC with precomputed kernel and evaluate."""
    print("\n► Training SVM with quantum kernel...")
    clf = SVC(kernel="precomputed", C=5.0, random_state=RANDOM_SEED)
    clf.fit(K_train, y_train)

    y_pred = clf.predict(K_test)
    y_prob = clf.decision_function(K_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = float("nan")

    print("\n" + "═"*55)
    print("  Q-IDS EVALUATION RESULTS")
    print("═"*55)
    print(f"  Accuracy          : {acc*100:.2f}%")
    print(f"  Weighted F1-Score : {f1*100:.2f}%")
    print(f"  AUC-ROC           : {auc:.4f}")
    print("─"*55)
    print(classification_report(y_test, y_pred, target_names=["Benign", "Attack"]))
    print("═"*55)

    return clf, y_pred, y_prob, acc, f1, auc


# ─────────────────────────────────────────────
# 5.  VISUALISATION
# ─────────────────────────────────────────────
def plot_results(K_train, y_train, y_test, y_pred, n_qubits, acc, f1, auc):
    """Generate and save result plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Q-IDS — {n_qubits}-Qubit Quantum Kernel SVM  |  "
        f"Acc {acc*100:.1f}%  F1 {f1*100:.1f}%  AUC {auc:.3f}",
        fontsize=13, fontweight="bold"
    )

    # ── Plot 1: Confusion Matrix ──────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Attack"],
        yticklabels=["Benign", "Attack"],
        ax=axes[0], linewidths=0.5
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # ── Plot 2: Quantum Kernel Matrix (train, first 80 samples) ──
    n_show = min(80, len(K_train))
    idx    = np.argsort(y_train)[:n_show]
    K_show = K_train[np.ix_(idx, idx)]
    im = axes[1].imshow(K_show, cmap="viridis", aspect="auto")
    axes[1].set_title(f"Quantum Kernel Matrix (first {n_show} train samples,\nsorted by class)")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Sample index")
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    # ── Plot 3: Per-class Precision / Recall / F1 bar chart ──────
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred)
    x = np.arange(2)
    width = 0.25
    axes[2].bar(x - width, p,     width, label="Precision", color="#4C72B0")
    axes[2].bar(x,         r,     width, label="Recall",    color="#DD8452")
    axes[2].bar(x + width, f,     width, label="F1",        color="#55A868")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["Benign", "Attack"])
    axes[2].set_ylim(0, 1.1)
    axes[2].set_title("Per-class Metrics")
    axes[2].legend()
    axes[2].set_ylabel("Score")

    plt.tight_layout()
    out = f"qids_{n_qubits}qubit_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n► Plot saved → {out}")
    plt.close()


# ─────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────
def main(n_qubits: int = 4):
    print("="*55)
    print(f"  Quantum IDS  |  {n_qubits} qubits  |  UNSW-NB15")
    print("="*55)

    # Load
    df = load_data(TRAIN_CSV, TEST_CSV)

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess(df, n_qubits)

    # Build quantum kernel
    print(f"\n► Building {n_qubits}-qubit quantum kernel circuit...")
    kernel_fn = build_kernel_circuit(n_qubits)

    # Compute kernel matrices
    print(f"\n► Computing kernel matrices  "
          f"(train: {len(X_train)}×{len(X_train)}, test: {len(X_test)}×{len(X_train)})...")
    print("  This is the slow step — each entry requires one circuit evaluation.")
    K_train = compute_kernel_matrix(X_train, X_train, kernel_fn, "Train")
    K_test  = compute_kernel_matrix(X_test,  X_train, kernel_fn, "Test ")

    # Train + evaluate
    clf, y_pred, y_prob, acc, f1, auc = train_and_evaluate(
        K_train, K_test, y_train, y_test
    )

    # Visualise
    plot_results(K_train, y_train, y_test, y_pred, n_qubits, acc, f1, auc)

    print("\n✓ Done.")
    return clf, acc, f1, auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum IDS — UNSW-NB15")
    parser.add_argument(
        "--qubits", type=int, default=4, choices=[2, 4],
        help="Number of qubits (2=fast, 4=more expressive). Default: 4"
    )
    args = parser.parse_args()
    main(n_qubits=args.qubits)