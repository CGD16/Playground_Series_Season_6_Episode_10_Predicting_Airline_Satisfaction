import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, roc_auc_score

from src import config as cfg


def rank01(a):
    """Konvertiert ein Array von Vorhersagen in Werte zwischen 0 und 1 basierend auf ihrem Rang."""
    return rankdata(a) / len(a)


def find_best_threshold(y_true, oof_preds):
    """Sucht den optimalen Schwellenwert (Threshold) für Accuracy auf gerankten oder ungerankten OOF-Daten."""
    best_thresh = 0.5
    best_acc = 0.0
    # Da Rank-01 Werte zwischen 0 und 1 liefert, deckt 0.3 bis 0.7 den relevanten Bereich ab:
    for thresh in np.linspace(0.3, 0.7, 401):
        acc = accuracy_score(y_true, (oof_preds >= thresh).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    return best_thresh, best_acc


def blend(results: dict, y):
    """results: {name: {'oof': ..., 'test': ...}} -> (Test-Binär, Test-Prob-Ranked, OOF-AUC, best_thresh, best_acc)."""
    names = list(results)

    # 1. Vorhersagen in Prozentränge (0..1) umwandeln
    oof_r = np.column_stack([rank01(results[n]["oof"]) for n in names])
    test_r = np.column_stack([rank01(results[n]["test"]) for n in names])

    # 2. Auswertung mit gleichen Gewichten
    equal_w = np.ones(len(names)) / len(names)
    equal_auc = roc_auc_score(y, oof_r @ equal_w)
    print(f"Rank-Average (gleiche Gewichte):    {equal_auc:.6f}")

    def norm(w):
        w = np.abs(w)
        return w / w.sum()

    # 3. Optimierung der Gewichte für maximale AUC
    res = minimize(
        lambda w: -roc_auc_score(y, oof_r @ norm(w)),
        equal_w,
        method="Nelder-Mead",
        options={"maxiter": 200},
    )
    w_opt = norm(res.x)
    opt_auc = roc_auc_score(y, oof_r @ w_opt)
    print(
        f"Rank-Average (optimierte Gewichte): {opt_auc:.6f}  "
        f"{dict(zip(names, [round(float(x), 3) for x in w_opt]))}"
    )

    # 4. Gewichte auswählen
    if opt_auc - equal_auc < 1e-5:
        final_oof = oof_r @ equal_w
        final_test_prob = test_r @ equal_w
        final_auc = equal_auc
    else:
        final_oof = oof_r @ w_opt
        final_test_prob = test_r @ w_opt
        final_auc = opt_auc

    # 5. Optimalen Schwellenwert auf den gerankten OOF-Daten finden
    best_thresh, best_acc = find_best_threshold(y, final_oof)
    print(
        f"Optimaler Rang-Schwellenwert: {best_thresh:.4f} | OOF Accuracy: {best_acc:.6f}"
    )

    # 6. Finale binäre Vorhersagen (0 oder 1) für das Testset erzeugen
    final_test_binary = (final_test_prob >= best_thresh).astype(int)

    return final_test_binary, final_test_prob, final_auc, best_thresh, best_acc