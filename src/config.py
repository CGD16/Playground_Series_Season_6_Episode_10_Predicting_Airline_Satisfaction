# (Pfade, Seeds, N_SPLITS, welche Modelle/Feature-Sets)

"""Zentrale Konfiguration - HIER werden Experimente gesteuert."""
from pathlib import Path

# --- Pfade (relativ zum Projektordner, unabhaengig davon, wo du startest) ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Dataset"
MODELS_DIR = ROOT / "Models"
RESULTS_DIR = ROOT / "Results"
SUBMISSIONS_DIR = RESULTS_DIR / "submissions"
EXPERIMENTS_LOG = RESULTS_DIR / "experiments.csv"

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SAMPLE_SUB_FILE = "sample_submission.csv"
ORIGINAL_FILE = "original.csv"  # z.B. "original.csv" (liegt in Dataset/); None = nicht verwenden

TARGET = "satisfaction"
ID_COL = "id"

# --- Experiment ---
# EXP_NAME = "exp001_baseline"       # wird in Dateinamen und Log verwendet
# EXP_NAME = "exp002_route_features"
# FEATURE_SET = "fs03"               # fs00 = roh, fs01 = Basis-FE, fs02 = fs01 + Rating-Aggregate
# FEATURE_SET = "fs05"
# EXP_NAME = "exp005_advanced_features"
# FEATURE_SET = "fs06"
# EXP_NAME = "exp006_joint_stats"
FEATURE_SET = "fs07"
EXP_NAME = "exp007_teacher_no_orig_rows"

# --- Original-Daten-Schalter (Ergebnisse der Ablation: v3 / v8) ---
USE_ORIG_ROWS = False    # False: Original-Zeilen NICHT ins Training haengen (Teacher-Feature bleibt!)
ORIG_IN_STATS = True    # False: Aggregate nur auf Train+Test, Original bekommt eigene Statistiken
# Suffix fuer Checkpoint-/OOF-Dateinamen, damit Varianten sich nicht gegenseitig ueberschreiben.
# Bei den Standardwerten (True/True) bleibt alles wie bisher -> bestehende Checkpoints gelten weiter.
VARIANT_SUFFIX = ("" if USE_ORIG_ROWS else "_noOrigRows") + ("" if ORIG_IN_STATS else "_origSep")

MODELS = ["xgb", "lgbm", "cat"]    # beliebig kombinierbar
N_SPLITS = 10
SEEDS = [42, 2026, 10, 23, 16, 2, 8, 46] 
USE_GPU = True                     # True bei NVIDIA-GPU mit CUDA
RUN_ADVERSARIAL = False            # Train-vs-Test-Check vor dem Training

ROUNDS = {"xgb": 8000, "lgbm": 10000, "cat": 8000}  # max. Baeume, Early Stopping greift vorher
EARLY_STOPPING = 200

VERBOSE = 1

# --- Modell-Parameter (hier tauschen/tunen) ---
PARAMS = {
    "xgb": dict(
        learning_rate=0.009,
        max_depth=9,
        min_child_weight=7,
        subsample=0.8593270986225269,
        colsample_bytree=0.6616735383572131,
        gamma=0.6136611655414813,
        reg_alpha=0.009239335165807,
        reg_lambda=3.6009252105250233,
        tree_method="hist", 
        device="cuda",       
    ),
    "lgbm": dict(
        learning_rate=0.007,
        num_leaves=128,
        max_depth=10,
        min_child_samples=100,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        device="cuda",      # cuda oder gpu 
    ),
    "cat": dict(
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3.0,
        task_type="GPU", 
        metric_period=10, # 10 ist schneller als 5
    ),
}

# --- Feature-Definitionen ---
ORDINAL_FEATURES = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness",
]
CLASS_MAP = {"Eco": 0, "Eco Plus": 1, "Business": 2}