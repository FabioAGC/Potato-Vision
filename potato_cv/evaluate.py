"""Métricas: classificador sozinho, PaDiM sozinho e o sistema híbrido."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from .decision import FOREIGN, OK, UNKNOWN, Decision


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Intervalo de confiança 95% (Wilson) para uma proporção k/n."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0, c - h)), float(min(1, c + h))


def compute_metrics(y_true: list[str], probs: np.ndarray, scores: np.ndarray, decisions: list[Decision],
                    names: list[str], normal: set[str], threshold: float) -> dict:
    y_true = np.array(y_true)
    is_foreign = np.array([c not in normal for c in y_true])
    n_f, n_n = int(is_foreign.sum()), int((~is_foreign).sum())

    # ---- classificador sozinho
    pred = np.array([names[i] for i in probs.argmax(1)])
    cls_alarm = np.array([p not in normal for p in pred])
    labels = sorted(set(names) | set(y_true))
    cm = confusion_matrix(y_true, pred, labels=labels)

    # ---- PaDiM sozinho
    pad_alarm = scores >= threshold
    auroc = float(roc_auc_score(is_foreign, scores)) if 0 < n_f < len(y_true) else None

    # ---- híbrido
    status = np.array([d.status for d in decisions])
    hyb_alarm = status != OK

    def binary(alarm):
        tp = int((alarm & is_foreign).sum())
        fp = int((alarm & ~is_foreign).sum())
        return {
            "recall_corpo_estranho": _rate(tp, n_f),
            "recall_ic95": wilson(tp, n_f),
            "falso_alarme_batata": _rate(fp, n_n),
            "falso_alarme_ic95": wilson(fp, n_n),
            "precisao": _rate(tp, tp + fp),
            "detectados": tp, "perdidos": n_f - tp, "falsos_alarmes": fp,
        }

    per_class = {}
    for c in labels:
        m = y_true == c
        if not m.any():
            continue
        per_class[c] = {
            "n": int(m.sum()),
            "classificador_acerto": float((pred[m] == c).mean()),
            "hibrido_alarme": float(hyb_alarm[m].mean()),
            "hibrido_foreign": float((status[m] == FOREIGN).mean()),
            "hibrido_unknown": float((status[m] == UNKNOWN).mean()),
            "padim_alarme": float(pad_alarm[m].mean()),
        }

    return {
        "n_imagens": len(y_true), "n_corpos_estranhos": n_f, "n_batatas": n_n,
        "classificador": {"acuracia_4_classes": float((pred == y_true).mean()), **binary(cls_alarm),
                          "matriz_confusao": {"labels": labels, "matriz": cm.tolist()}},
        "padim": {"auroc": auroc, "limiar": threshold, **binary(pad_alarm)},
        "hibrido": binary(hyb_alarm),
        "por_classe": per_class,
    }


def fmt_pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def fmt_ci(ci) -> str:
    return "" if not ci else f" ({100 * ci[0]:.0f}–{100 * ci[1]:.0f}%)"
