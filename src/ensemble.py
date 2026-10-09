import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from src import config as cfg


def rank01(a):
    """Konvertiert ein Array von Vorhersagen in Werte zwischen 0 und 1 basierend auf ihrem Rang."""
    return rankdata(a) / len(a)


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blend(results: dict, y, mode: str = "prob"):
    """Blendet die Modelle zu WAHRSCHEINLICHKEITEN (0..1), keine Binaerwerte.

    results: {name: {'oof': ..., 'test': ...}}
    mode:
      "prob"  - gewichteter Mittelwert der Roh-Wahrscheinlichkeiten (Ausgabe = echte Probabilities)
      "logit" - gewichteter Mittelwert im Logit-Raum, danach Sigmoid (Ausgabe = echte Probabilities)
      "rank"  - Rank-Average (Werte 0..1, aber keine kalibrierten Probabilities)

    Rueckgabe: (test_prob, oof_prob, auc)
    """
    names = list(results)

    if mode == "prob":
        to_space = lambda a: np.asarray(a, dtype=np.float64)
        back = lambda z: z
    elif mode == "logit":
        to_space = _logit
        back = _sigmoid
    elif mode == "rank":
        to_space = rank01
        back = lambda z: z
    else:
        raise ValueError(f"Unbekannter mode: {mode}")

    oof_m = np.column_stack([to_space(results[n]["oof"]) for n in names])
    test_m = np.column_stack([to_space(results[n]["test"]) for n in names])

    equal_w = np.ones(len(names)) / len(names)
    equal_auc = roc_auc_score(y, oof_m @ equal_w)
    print(f"Blend [{mode}] (gleiche Gewichte):    {equal_auc:.6f}")

    def norm(w):
        w = np.abs(w)
        return w / w.sum()

    res = minimize(
        lambda w: -roc_auc_score(y, oof_m @ norm(w)),
        equal_w,
        method="Nelder-Mead",
        options={"maxiter": 200},
    )
    w_opt = norm(res.x)
    opt_auc = roc_auc_score(y, oof_m @ w_opt)
    print(
        f"Blend [{mode}] (optimierte Gewichte): {opt_auc:.6f}  "
        f"{dict(zip(names, [round(float(x), 3) for x in w_opt]))}"
    )

    # Optimierte Gewichte nur nehmen, wenn sie spuerbar besser sind (sonst Overfitting-Gefahr)
    w = w_opt if opt_auc - equal_auc >= 1e-5 else equal_w
    final_auc = max(opt_auc, equal_auc) if w is w_opt else equal_auc

    final_oof = np.clip(back(oof_m @ w), 0.0, 1.0)
    final_test = np.clip(back(test_m @ w), 0.0, 1.0)

    return final_test, final_oof, final_auc