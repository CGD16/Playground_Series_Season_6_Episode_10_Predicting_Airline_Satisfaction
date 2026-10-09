# (Feature-Sets: fs01 ... fs06)
"""Feature-Sets. Neues Set: Funktionen kombinieren und in FEATURE_SETS eintragen."""
import numpy as np
import pandas as pd
from src import config as cfg


def _delay(df):
    df["Arrival_Delay_Missing"] = df["Arrival Delay in Minutes"].isna().astype("int8")
    df["Delay_Difference"] = df["Arrival Delay in Minutes"] - df["Departure Delay in Minutes"]


def _combos(df):
    cls = df["Class"].astype(str)
    trv = df["Type of Travel"].astype(str)
    cust = df["Customer Type"].astype(str)
    df["Class_TypeTravel"] = cls + "_" + trv
    df["Class_CustomerType"] = cls + "_" + cust
    df["Class_TypeTravel_CustomerType"] = cls + "_" + trv + "_" + cust
    df["Class_Wifi"] = cls + "_" + df["Inflight wifi service"].astype(str)


def _distance(df):
    df["Distance_Class"] = df["Flight Distance"] * df["Class"].map(cfg.CLASS_MAP)
    bins = [-np.inf, 500, 1000, 1200, 1500, 1800, 2500, 3500, np.inf]
    df["Distance_Bin"] = pd.cut(df["Flight Distance"], bins=bins, labels=False).astype(str)


def _rating_agg(df):
    r = df[cfg.ORDINAL_FEATURES]
    df["Rating_Mean"] = r.mean(axis=1)
    df["Rating_Min"] = r.min(axis=1)
    df["Rating_Max"] = r.max(axis=1)
    df["Rating_Std"] = r.std(axis=1)
    df["N_Zero"] = (r == 0).sum(axis=1)
    df["N_Low"] = r.isin([1, 2]).sum(axis=1)
    df["N_High"] = (r >= 4).sum(axis=1)


# --- ERWEITERTE RATING-STATISTIKEN (FE-Hebel 1) ---
def _rating_agg_extended(df):
    r = df[cfg.ORDINAL_FEATURES]
    df["Rating_Mean"] = r.mean(axis=1)
    df["Rating_Std"] = r.std(axis=1)
    df["Rating_Min"] = r.min(axis=1)
    df["Rating_Max"] = r.max(axis=1)
    df["Rating_Sum"] = r.sum(axis=1)
    df["Rating_Skew"] = r.skew(axis=1)

    # Häufigkeits-Aggregate über die Bewertungsstufen
    df["N_Zero"] = (r == 0).sum(axis=1)
    df["N_Low"] = r.isin([1, 2]).sum(axis=1)
    df["N_High"] = (r >= 4).sum(axis=1)


def _impute_clean(df):
    """Fehlende & negative Werte bereinigen."""
    if "Arrival Delay in Minutes" in df.columns and "Departure Delay in Minutes" in df.columns:
        df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])

    cont_cols = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
    for c in cont_cols:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan


def _route_features(df):
    """Flight Distance als eindeutige Route_ID behandeln und Aggregationen ableiten.
    Wird auf dem gemeinsamen Frame (Train+Test+Orig) berechnet -> einheitliche Skala."""
    df["Route_ID"] = df["Flight Distance"].fillna(-1).astype(int)

    freq_map = df["Route_ID"].value_counts()
    df["Route_Freq"] = df["Route_ID"].map(freq_map).astype(np.int32)
    df["Route_Freq_Ratio"] = df["Route_Freq"] / len(df)

    freq_q75 = freq_map.quantile(0.75)
    df["Route_Is_Common"] = (df["Route_Freq"] >= freq_q75).astype("int8")

    profile_cols = [c for c in ["Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"] if c in df.columns]
    profile_cols += [c for c in cfg.ORDINAL_FEATURES if c in df.columns]

    for col in profile_cols:
        mean_map = df.groupby("Route_ID")[col].transform("mean")
        col_clean = col.replace(" ", "_").replace("/", "_")
        df[f"Route_Mean_{col_clean}"] = mean_map


def _aux_expected_ratings(df):
    """Berechnet die Abweichungen (Residuen) vom erwarteten Gruppen-Rating."""
    segment = df["Class"].astype(str) + "_" + df["Type of Travel"].astype(str) + "_" + df["Customer Type"].astype(str)

    for col in cfg.ORDINAL_FEATURES:
        if col in df.columns:
            global_mean = df[col].mean()
            mean_map = df.groupby(segment)[col].transform("mean").fillna(global_mean)
            col_clean = col.replace(" ", "_").replace("/", "_")

            df[f"Exp_Rating_{col_clean}"] = mean_map
            df[f"Diff_Exp_Rating_{col_clean}"] = df[col] - mean_map


# --- INTERAKTIONEN (FE-Hebel 2) ---
def _interactions(df):
    if "Online boarding" in df.columns and "Inflight wifi service" in df.columns:
        df["Wifi_x_Boarding"] = df["Inflight wifi service"] * df["Online boarding"]

    if "Arrival Delay in Minutes" in df.columns and "Departure Delay in Minutes" in df.columns:
        df["Total_Delay"] = df["Arrival Delay in Minutes"] + df["Departure Delay in Minutes"]
        df["Delay_Ratio"] = df["Arrival Delay in Minutes"] / (df["Departure Delay in Minutes"] + 1)


# --- GROUP AGGREGATIONS (FE-Hebel 3) ---
def _group_aggregations(df):
    if all(c in df.columns for c in ["Class", "Customer Type", "Flight Distance"]):
        group_cols = ["Class", "Customer Type"]
        mean_dist = df.groupby(group_cols)["Flight Distance"].transform("mean")
        df["Group_Dist_Mean"] = mean_dist
        df["Group_Dist_Std"] = df.groupby(group_cols)["Flight Distance"].transform("std")
        df["Diff_From_Group_Dist_Mean"] = df["Flight Distance"] - mean_dist

    if "Class" in df.columns and "Departure Delay in Minutes" in df.columns:
        df["Class_DepDelay_Mean"] = df.groupby("Class")["Departure Delay in Minutes"].transform("mean")


_FS05 = [
    _impute_clean,
    _delay,
    _combos,
    _distance,
    _rating_agg_extended,
    _route_features,
    _aux_expected_ratings,
    _interactions,
    _group_aggregations,
]

FEATURE_SETS = {
    "fs00": [],                                          # Rohdaten
    "fs01": [_delay, _combos, _distance],                # Basis-FE (bereinigt)
    "fs02": [_delay, _combos, _distance, _rating_agg],   # + Rating-Aggregate
    "fs03": [_impute_clean, _delay, _combos, _distance, _rating_agg, _route_features],  # + Routen-Features
    "fs04": [_impute_clean, _delay, _combos, _distance, _rating_agg, _route_features, _aux_expected_ratings],
    "fs05": _FS05,                                       # Erweitertes FE-Set (Stats pro DataFrame)
    "fs06": _FS05,                                       # Wie fs05, aber via add_features_joint (gemeinsame Stats)
}


def add_features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    df = df.copy()
    for fn in FEATURE_SETS[feature_set]:
        fn(df)
    return df


def add_features_joint(train, test, orig, feature_set, orig_in_stats=True):
    """Berechnet alle Aggregate/Haeufigkeiten EINMAL auf concat(train, test[, orig])
    und teilt danach wieder auf. Label-frei -> kein Leakage, aber identische
    Skala/Rauschniveau fuer Train, Test und Original.

    orig_in_stats=False: Original wird separat (eigene Statistik) berechnet.
    """
    parts = [train, test]
    if orig is not None and orig_in_stats:
        parts.append(orig)
    sizes = [len(p) for p in parts]

    full = add_features(pd.concat(parts, ignore_index=True), feature_set)

    out, start = [], 0
    for n in sizes:
        out.append(full.iloc[start:start + n].reset_index(drop=True))
        start += n

    X_tr, X_te = out[0], out[1]
    if orig is None:
        X_or = None
    elif orig_in_stats:
        X_or = out[2]
    else:
        X_or = add_features(orig, feature_set).reset_index(drop=True)
    return X_tr, X_te, X_or