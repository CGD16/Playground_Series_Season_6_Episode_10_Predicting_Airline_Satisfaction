# data.py

#  (Laden, Originaldatensatz, Kategorien)
"""Daten laden und fuer die Modelle vorbereiten."""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from src import config as cfg
from src.features import add_features, add_features_joint
 

import lightgbm as lgb
import xgboost as xgb  


@dataclass
class Data:
    X: pd.DataFrame
    y: pd.Series
    X_test: pd.DataFrame
    X_orig: Optional[pd.DataFrame]
    y_orig: Optional[pd.Series]
    cat_cols: List[str]
    test: pd.DataFrame         
    sample_sub: pd.DataFrame


def to_binary(s: pd.Series) -> pd.Series:
    """Target als 0/1 (Zahlen bleiben, Text: 'satisfied' = 1)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)
    return (s.astype(str).str.strip().str.lower() == "satisfied").astype(int)


def add_route_target_encoding(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, X_orig: Optional[pd.DataFrame] = None, n_folds=5, smoothing=20.0, seed=42):
    """Berechnet out-of-fold smoothed Target-Encoding auf der Route_ID."""
    train_encoded = np.full(len(X), np.nan, dtype=np.float32)
    global_mean = y.astype(float).mean()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, va_idx in kf.split(X):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        df_tr = pd.DataFrame({"Route_ID": X_tr["Route_ID"], "target": y_tr})
        
        agg = df_tr.groupby("Route_ID")["target"].agg(["mean", "count"])
        smoothed = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
        mapping = smoothed.to_dict()
        
        train_encoded[va_idx] = X.iloc[va_idx]["Route_ID"].map(mapping).fillna(global_mean).values

    df_all_tr = pd.DataFrame({"Route_ID": X["Route_ID"], "target": y})
    agg = df_all_tr.groupby("Route_ID")["target"].agg(["mean", "count"])
    smoothed = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
    mapping = smoothed.to_dict()

    test_encoded = X_test["Route_ID"].map(mapping).fillna(global_mean).values.astype(np.float32)

    orig_encoded = None
    if X_orig is not None:
        orig_encoded = X_orig["Route_ID"].map(mapping).fillna(global_mean).values.astype(np.float32)

    return train_encoded, test_encoded, orig_encoded


def load_data(feature_set: str) -> Data:
    train = pd.read_csv(cfg.DATA_DIR / cfg.TRAIN_FILE)
    test = pd.read_csv(cfg.DATA_DIR / cfg.TEST_FILE)
    sample_sub = pd.read_csv(cfg.DATA_DIR / cfg.SAMPLE_SUB_FILE)

    orig = None
    if cfg.ORIGINAL_FILE is not None:
        path = cfg.DATA_DIR / cfg.ORIGINAL_FILE
        if path.exists():
            orig = pd.read_csv(path)
            keep = [c for c in train.columns if c in orig.columns and c != cfg.ID_COL]
            orig = orig[keep].copy()
            print(f"Originaldatensatz geladen: {orig.shape}")
        else:
            print(f"WARNUNG: {path} nicht gefunden - Originaldaten werden ignoriert")

    print(f"train {train.shape} | test {test.shape} | sample_submission {sample_sub.shape}")

    y = to_binary(train[cfg.TARGET]).reset_index(drop=True)
    X_raw = train.drop(columns=[cfg.TARGET, cfg.ID_COL])
    X_test_raw = test.drop(columns=[cfg.ID_COL])
    X_orig_raw, y_orig = None, None
    if orig is not None:
        y_orig = to_binary(orig[cfg.TARGET]).reset_index(drop=True)
        X_orig_raw = orig.drop(columns=[cfg.TARGET])

    # Aggregate/Haeufigkeiten einmal auf Train+Test(+Orig) berechnen (label-frei, kein Leakage)
    # cfg.ORIG_IN_STATS=False: Original wird NICHT in die Statistiken von Train/Test einbezogen
    X, X_test, X_orig = add_features_joint(
        X_raw, X_test_raw, X_orig_raw, feature_set, orig_in_stats=cfg.ORIG_IN_STATS
    )
    if X_orig is not None:
        X_orig = X_orig[X.columns]

    # --- INLINE-LAMBDA: Alle Leerzeichen in Spaltennamen durch Unterstriche ersetzen ---
    X, X_test, X_orig = [df.rename(columns=lambda c: c.replace(" ", "_")) if df is not None else None for df in (X, X_test, X_orig)]

    # --- AUSKOMMENTIERT: Globales Encoding entfernt, da es nun in-fold in src/cv.py läuft ---
    # if "Route_ID" in X.columns:
    #     enc_tr, enc_te, enc_orig = add_route_target_encoding(X, y, X_test, X_orig)
    #     X["Route_Target_Enc"] = enc_tr
    #     X_test["Route_Target_Enc"] = enc_te
    #     if X_orig is not None:
    #         X_orig["Route_Target_Enc"] = enc_orig
    #     print("Route_Target_Enc erfolgreich hinzugefuegt!")

    # --- SCHRITT 1: Kategoriale Spalten VOR dem Teacher stabil als Pandas "category" definieren ---
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    frames = [d for d in (X, X_test, X_orig) if d is not None]
    for c in cat_cols:
        cats = sorted(set().union(*[set(d[c].astype(str).unique()) for d in frames]))
        for d in frames:
            d[c] = pd.Categorical(d[c].astype(str), categories=cats)

    # --- SCHRITT 2: ORIGINAL-DATA TEACHER FEATURE (Mit nativer Category-Unterstützung) ---
    if orig is not None:
        print("Trainiere Original-Data Teacher Modell...")
        
        # Spalten für das Modell filtern
        feat_cols = [c for c in X.columns if c != "Route_ID" and c != "Route_Target_Enc"]
        teacher_feats = [c for c in feat_cols if c in X_orig.columns]

        # Logit-Hilfsfunktion (Wahrscheinlichkeiten stabil in Logits wandeln)
        logit = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))

        # Teacher OUT-OF-FOLD auf Original trainieren:
        #  - Original-Zeilen bekommen ihre Logits nur von Modellen, die sie NICHT gesehen haben
        #    (sonst sind die Logits auf Original in-sample ~perfekt, auf Train/Test aber nicht -> Verteilungsshift)
        #  - Train/Test-Logits = Mittel der Fold-Teacher (alle out-of-sample)
        teacher_kwargs = dict(
            n_estimators=600, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.5, tree_method="hist",
            enable_categorical=True,  # Zwingend erforderlich fuer Pandas Kategorien in XGBoost
            device="cuda" if cfg.USE_GPU else "cpu",
        )
        n_t = 5
        orig_oof = np.zeros(len(X_orig), dtype=np.float64)
        p_train = np.zeros(len(X), dtype=np.float64)
        p_test = np.zeros(len(X_test), dtype=np.float64)

        t_skf = StratifiedKFold(n_splits=n_t, shuffle=True, random_state=42)
        for t_tr, t_va in t_skf.split(X_orig, y_orig):
            tm = xgb.XGBClassifier(**teacher_kwargs, random_state=42)
            tm.fit(X_orig.iloc[t_tr][teacher_feats], y_orig.iloc[t_tr])
            orig_oof[t_va] = tm.predict_proba(X_orig.iloc[t_va][teacher_feats])[:, 1]
            p_train += tm.predict_proba(X[teacher_feats])[:, 1] / n_t
            p_test += tm.predict_proba(X_test[teacher_feats])[:, 1] / n_t

        print(f"Teacher OOF-AUC auf Original: {roc_auc_score(y_orig, orig_oof):.6f}")

        X["original_teacher_logit"] = logit(p_train)
        X_test["original_teacher_logit"] = logit(p_test)
        X_orig["original_teacher_logit"] = logit(orig_oof)
        print("Original-Data Teacher Feature erfolgreich generiert!")

    flag = "Arrival_Delay_Missing"
    if flag in X_test.columns and X_test[flag].sum() == 0:
        print(f"Hinweis: Test hat keine fehlenden Arrival-Delay-Werte -> '{flag}' entfernt")
        X = X.drop(columns=[flag])
        X_test = X_test.drop(columns=[flag])
        if X_orig is not None:
            X_orig = X_orig.drop(columns=[flag])

    print(f"Feature-Set {feature_set}: {X.shape[1]} Features (kategorisch: {len(cat_cols)})")
    
    return Data(X, y, X_test, X_orig, y_orig, cat_cols, test, sample_sub)


def adversarial_validation(X: pd.DataFrame, X_test: pd.DataFrame) -> float:
    """AUC ~0.50 = Train und Test sind gleich verteilt."""
    data = pd.concat([X, X_test], ignore_index=True)
    label = np.r_[np.zeros(len(X)), np.ones(len(X_test))]
    oof = np.zeros(len(data))
    for tr, va in StratifiedKFold(cfg.N_SPLITS, shuffle=True, random_state=0).split(data, label):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, verbose=-1)
        m.fit(data.iloc[tr], label[tr])
        oof[va] = m.predict_proba(data.iloc[va])[:, 1]
    auc = roc_auc_score(label, oof)
    print(f"Adversarial-AUC: {auc:.4f}")
    return auc