# (XGB/LGBM/CatBoost inkl. Parameter)
"""Modelle. Neues Modell: Zweig in fit_predict() ergaenzen + Parameter in config.PARAMS."""
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from src import config as cfg
import gc


def fit_predict(name, X_tr, y_tr, X_va, y_va, X_te, cat_cols, seed, fold=1):
    """Trainiert ein Modell -> (Valid-Vorhersage, Test-Vorhersage, beste Iteration).

    Speichert das trainierte Modell sowie die Vorhersage-Checkpoints in Models/saved_models/.
    Falls bereits Vorhersagen existieren, werden diese direkt geladen.
    """

    # Ordnerpfad zum Speichern vorbereiten
    save_dir = cfg.MODELS_DIR / "saved_models" / f"{name}_{cfg.FEATURE_SET}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Dateipfade fuer Vorhersage-Checkpoints
    pred_va_file = save_dir / f"pred_va_{name}_seed{seed}_fold{fold}.npy"
    pred_te_file = save_dir / f"pred_te_{name}_seed{seed}_fold{fold}.npy"

    # --- CHECKPOINT-CHECK: Falls Vorhersagen schon existieren, direkt laden ---
    if pred_va_file.exists() and pred_te_file.exists():
        va_pred = np.load(pred_va_file)
        te_pred = np.load(pred_te_file)
        return va_pred, te_pred, 0

    model_filename = save_dir / f"{name}_seed{seed}_fold{fold}"

    if name == "xgb":
        model = xgb.XGBClassifier(
            **cfg.PARAMS["xgb"],
            n_estimators=cfg.ROUNDS["xgb"],
            objective="binary:logistic",
            eval_metric="auc",
            enable_categorical=True,
            early_stopping_rounds=cfg.EARLY_STOPPING,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        best_it = model.best_iteration
        # Speichert das Modell im nativen JSON-Format
        model.save_model(f"{model_filename}.json")

    elif name == "lgbm":
        model = lgb.LGBMClassifier(
            **cfg.PARAMS["lgbm"],
            n_estimators=cfg.ROUNDS["lgbm"],
            random_state=seed,
            verbosity=-1,
            n_jobs=-1,
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(cfg.EARLY_STOPPING, verbose=False)],
        )
        best_it = model.best_iteration_
        # Speichert den Booster im nativen Textformat
        model.booster_.save_model(f"{model_filename}.txt")

    elif name == "cat":
        as_str = {c: str for c in cat_cols}
        X_tr, X_va, X_te = [d.astype(as_str) for d in (X_tr, X_va, X_te)]

        model = CatBoostClassifier(
            **cfg.PARAMS["cat"],
            iterations=cfg.ROUNDS["cat"],
            loss_function="Logloss",
            eval_metric="AUC",
            early_stopping_rounds=cfg.EARLY_STOPPING,
            random_seed=seed,
            cat_features=cat_cols,
        )
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        best_it = model.get_best_iteration()
        # Speichert das Modell im nativen CBM-Format
        model.save_model(f"{model_filename}.cbm")

    else:
        raise ValueError(f"Unbekanntes Modell: {name}")

    # Vorhersagen berechnen
    va_pred = model.predict_proba(X_va)[:, 1]
    te_pred = model.predict_proba(X_te)[:, 1]

    # --- CHECKPOINT SPEICHERN ---
    np.save(pred_va_file, va_pred)
    np.save(pred_te_file, te_pred)

    # Modell-Speicher freigeben, um Abstürze durch RAM/VRAM-Mangel zu verhindern
    del model
    gc.collect()

    return va_pred, te_pred, best_it