"""Fusão das duas opiniões (classificador + PaDiM) em uma decisão por objeto.

    FOREIGN  - corpo estranho conhecido (ex.: Pedra, Planta)   -> vermelho
    UNKNOWN  - PaDiM acha anormal e o classificador não tem
               certeza de que é batata                         -> amarelo
    OK       - batata (boa ou danificada)                      -> sem alerta
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OK, FOREIGN, UNKNOWN = "ok", "foreign", "unknown"


@dataclass
class Decision:
    status: str          # ok | foreign | unknown
    label: str           # classe prevista ou "Unknown"
    confidence: float    # prob. do classificador para o rótulo
    anomaly: float       # nota PaDiM normalizada (1.0 = no limiar)


def decide(probs: np.ndarray, anomaly: np.ndarray, names: list[str], normal_classes: set[str],
           threshold: float, foreign_conf: float, normal_conf: float) -> list[Decision]:
    """probs (N, C), anomaly (N,) bruto do PaDiM."""
    normal_idx = np.array([n in normal_classes for n in names])
    out = []
    for p, a in zip(probs, anomaly):
        a_norm = float(a / threshold) if threshold else 0.0
        top = int(p.argmax())
        p_normal = float(p[normal_idx].sum())
        if not normal_idx[top] and p[top] >= foreign_conf:
            # 1) classificador reconhece um corpo estranho conhecido
            out.append(Decision(FOREIGN, names[top], float(p[top]), a_norm))
        elif a_norm >= 1.0 and p_normal < normal_conf:
            # 2) PaDiM acha anormal e o classificador não garante que é batata
            out.append(Decision(UNKNOWN, "Unknown", float(p[top]), a_norm))
        else:
            # 3) batata
            out.append(Decision(OK, names[top], float(p[top]), a_norm))
    return out
