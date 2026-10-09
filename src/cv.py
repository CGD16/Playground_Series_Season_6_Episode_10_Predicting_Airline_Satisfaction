# (Cross-Validation-Schleife)
"""Cross-Validation-Schleife. Speichert OOF-/Testvorhersagen in Models/ und loggt Ergebnisse."""
import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import config as cfg
from src.data import Data
from src.models import fit_predict


def run_tag(model: str, feature_set: str) -> str:
    """Eindeutiger Name, z.B. 'xgb_fs02_s42'."""
    return f"{model}_{feature_set}_s" + "-".join(str(s) for s in cfg.SEEDS)


def get_target_encode(k_train, y_train, k_valid, smooth=20.0):
    """Berechnet smoothed Target Encoding für Validierungs-/Testdaten basierend auf Trainingsstatistiken."""
    prior = y_train.mean()
    stats = pd.DataFrame({'k': k_train, 'y': y_train}).groupby('k')['y'].agg(['sum', 'count'])
    mapping = (stats['sum'] + prior * smooth) / (stats['count'] + smooth)
    return pd.Series(k_valid).map(mapping).fillna(prior).values.astype(np.float32)


def run_cv(name: str, data: Data, feature_set: str) -> dict:
    X, y = data.X.copy(), data.y
    
    # Array zum Sammeln aller OOF-Vorhersagen pro Seed (Form: [Anzahl_Seeds, Anzahl_Samples])
    oof_seeds = np.zeros((len(cfg.SEEDS), len(X)), dtype=np.float32)
    test_pred = np.zeros(len(data.X_test), dtype=np.float32)
    
    fold_times = []
    total_start_time = time.time()

    # Ordnerpfad zu den Modell-Checkpoints
    save_dir = cfg.MODELS_DIR / "saved_models" / f"{name}_{cfg.FEATURE_SET}"

    # Iteration über alle Seeds
    for s_idx, seed in enumerate(cfg.SEEDS):
        # temporäres OOF für den aktuellen Seed
        oof_single_seed = np.zeros(len(X), dtype=np.float32)
        
        skf = StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(X, y), 1):
            
            # Checkpoint-Pfade abfragen
            pred_va_file = save_dir / f"pred_va_{name}_seed{seed}_fold{fold}.npy"
            pred_te_file = save_dir / f"pred_te_{name}_seed{seed}_fold{fold}.npy"

            # Falls Checkpoint bereits existiert: Vorhersagen ohne Rechenaufwand laden
            if pred_va_file.exists() and pred_te_file.exists():
                va_pred = np.load(pred_va_file)
                te_pred = np.load(pred_te_file)
                best_it = 0
                fold_duration = 0.0
                print(f"[{name}] seed {seed} fold {fold}: "
                      f"geladen aus Checkpoint (AUC={roc_auc_score(y.iloc[va], va_pred):.6f})")
            else:
                X_tr, y_tr = X.iloc[tr].copy(), y.iloc[tr]
                X_va, y_va = X.iloc[va].copy(), y.iloc[va]
                X_te = data.X_test.copy()

                # --- ECHTES IN-FOLD TARGET ENCODING ---
                if "Route_ID" in X_tr.columns:
                    inner_oof = np.zeros(len(X_tr), dtype=np.float32)
                    inner_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                    
                    for inner_tr, inner_va in inner_kf.split(X_tr, y_tr):
                        inner_oof[inner_va] = get_target_encode(
                            X_tr["Route_ID"].iloc[inner_tr], 
                            y_tr.iloc[inner_tr], 
                            X_tr["Route_ID"].iloc[inner_va]
                        )
                    X_tr["Route_Target_Enc"] = inner_oof
                    
                    X_va["Route_Target_Enc"] = get_target_encode(X_tr["Route_ID"], y_tr, X_va["Route_ID"])
                    X_te["Route_Target_Enc"] = get_target_encode(X_tr["Route_ID"], y_tr, X_te["Route_ID"])

                if data.X_orig is not None:
                    X_o = data.X_orig.copy()
                    # Original-Zeilen bekommen das Encoding aus den Train-Fold-Labels (wie Valid/Test),
                    # sonst waeren sie NaN, waehrend alle anderen Trainingszeilen einen Wert haben
                    if "Route_ID" in X_tr.columns:
                        X_o["Route_Target_Enc"] = get_target_encode(X_tr["Route_ID"], y_tr, X_o["Route_ID"])
                    X_tr = pd.concat([X_tr, X_o], ignore_index=True)
                    y_tr = pd.concat([y_tr, data.y_orig], ignore_index=True)

                start_time = time.time()

                # Modelltraining & Checkpoint-Speicherung über fit_predict
                va_pred, te_pred, best_it = fit_predict(
                    name, X_tr, y_tr, X_va, y_va, X_te, data.cat_cols, seed, fold=fold
                )

                fold_duration = time.time() - start_time
                fold_times.append(fold_duration)

                print(f"[{name}] seed {seed} fold {fold}: "
                      f"AUC={roc_auc_score(y.iloc[va], va_pred):.6f}  "
                      f"best_iter={best_it}  "
                      f"time={fold_duration:.2f}s")

            # In-Fold Vorhersagen für den aktuellen Seed ablegen
            oof_single_seed[va] = va_pred
            
            # Test-Vorhersagen fortlaufend dividiert durch Gesamtzahl aller Modelle addieren
            test_pred += te_pred / (cfg.N_SPLITS * len(cfg.SEEDS))
        
        # OOF-Ergebnis des aktuellen Seeds im Gesamt-Array speichern
        oof_seeds[s_idx] = oof_single_seed
        print(f"--> [{name}] Seed {seed} fertig | Single-Seed OOF-AUC: {roc_auc_score(y, oof_single_seed):.6f}")

    # --- SEED AVERAGING FÜR OOF ---
    oof_averaged = np.mean(oof_seeds, axis=0)

    total_duration = time.time() - total_start_time
    auc = roc_auc_score(y, oof_averaged)
    print(f"==> [{name}] FINAL Seed-Averaged OOF-AUC: {auc:.6f} | Gesamtzeit: {total_duration:.2f}s\n")

    # Speichern unter dem gewohnten Tag
    tag = run_tag(name, feature_set)
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cfg.MODELS_DIR / f"oof_{tag}.npy", oof_averaged)
    np.save(cfg.MODELS_DIR / f"test_{tag}.npy", test_pred)

    # Logging...
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    mean_fold_time = round(np.mean(fold_times), 2) if fold_times else 0.0
    fold_times_str = "-".join([f"{t:.1f}" for t in fold_times]) if fold_times else "checkpoints-used"

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exp_name": cfg.EXP_NAME,
        "model": name,
        "feature_set": feature_set,
        "oof_auc": round(auc, 6),
        "n_splits": cfg.N_SPLITS,
        "seeds": str(cfg.SEEDS),
        "use_gpu": cfg.USE_GPU,
        "total_time_sec": round(total_duration, 2),
        "mean_fold_time_sec": mean_fold_time,
        "fold_times": fold_times_str
    }
    
    df_entry = pd.DataFrame([log_entry])
    
    if cfg.EXPERIMENTS_LOG.exists():
        df_entry.to_csv(cfg.EXPERIMENTS_LOG, mode='a', header=False, index=False)
    else:
        df_entry.to_csv(cfg.EXPERIMENTS_LOG, mode='w', header=True, index=False)

    return {"oof": oof_averaged, "test": test_pred, "auc": auc, "tag": tag}