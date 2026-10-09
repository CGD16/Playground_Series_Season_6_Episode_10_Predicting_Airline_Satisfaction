# (sample_submission befüllen)
"""Submission schreiben und Experimente protokollieren."""
from datetime import datetime
import pandas as pd
from src import config as cfg


def write_submission(test, sample_sub, test_pred, cv_auc: float):
    id_col, target_col = sample_sub.columns[0], sample_sub.columns[1]
    # Zuordnung ueber die ID -> unabhaengig von der Zeilenreihenfolge
    pred_by_id = pd.Series(test_pred, index=test[cfg.ID_COL].values)
    sub = sample_sub.copy()
    sub[target_col] = pred_by_id.reindex(sub[id_col].values).values

    assert sub[target_col].notna().all(), "Fehlende Vorhersagen - IDs passen nicht"
    assert len(sub) == len(test)
    assert sub[target_col].between(0, 1).all()

    cfg.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y%m%d_%H%M")
    path = cfg.SUBMISSIONS_DIR / f"sub_{date}_{cfg.EXP_NAME}_cv{cv_auc:.4f}.csv"
    sub.to_csv(path, index=False)
    print(f"Submission gespeichert: {path}")
    print(sub.head())
    return path


def log_experiment(model: str, cv_auc: float):
    """Haengt eine Zeile an Results/experiments.csv an (public_lb traegst du manuell nach)."""
    row = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
        "experiment": cfg.EXP_NAME,
        "model": model,
        "feature_set": cfg.FEATURE_SET,
        "seeds": "-".join(str(s) for s in cfg.SEEDS),
        "n_splits": cfg.N_SPLITS,
        "use_original": cfg.ORIGINAL_FILE is not None,
        "cv_auc": round(cv_auc, 6),
        "public_lb": "",
    }
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    new = not cfg.EXPERIMENTS_LOG.exists()
    pd.DataFrame([row]).to_csv(cfg.EXPERIMENTS_LOG, mode="a", header=new, index=False)