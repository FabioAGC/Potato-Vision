"""Detector no quadro inteiro da câmera.

O quadro é varrido com uma janela do tamanho de UM objeto (`inference.tile_px`)
com sobreposição. Cada janela passa pelos dois modelos (em lote) e pela regra
de decisão; janelas alarmadas vizinhas são unidas numa só caixa.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .classifier import Classifier
from .config import Config, resolve
from .decision import FOREIGN, OK, UNKNOWN, decide
from .padim import PaDiM


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # x, y, w, h
    status: str                     # foreign | unknown | ok
    label: str
    confidence: float
    anomaly: float


class FrameDetector:
    def __init__(self, cfg: Config):
        engine = cfg.inference.get("engine", "torch")
        cls_w = resolve(cfg.classifier.weights)
        if engine == "onnx" and cls_w.with_suffix(".onnx").exists():
            cls_w = cls_w.with_suffix(".onnx")
        self.classifier = Classifier(cls_w, cfg.classifier.imgsz)
        self.padim = PaDiM.load(resolve(cfg.padim.weights), engine=engine)
        self.normal = set(cfg.normal_classes)
        self.dec = cfg.decision
        inf = cfg.inference
        self.tile = int(inf.tile_px)
        self.stride = max(1, int(round(self.tile * float(inf.stride))))
        self.roi = inf.get("roi")
        self.mode = inf.get("aggregation", "grid")
        self.last_ms: dict[str, float] = {}

    # ------------------------------------------------------------ crops
    def classify_crops(self, crops: list[np.ndarray]):
        probs = self.classifier.predict(crops)
        scores = self.padim.score(crops)
        return decide(probs, scores, self.classifier.names, self.normal, self.padim.threshold,
                      self.dec.foreign_conf, self.dec.normal_conf)

    def _windows(self, w: int, h: int):
        t, s = self.tile, self.stride
        xs = list(range(0, max(1, w - t + 1), s))
        ys = list(range(0, max(1, h - t + 1), s))
        if xs[-1] + t < w:
            xs.append(max(0, w - t))
        if ys[-1] + t < h:
            ys.append(max(0, h - t))
        return [(x, y) for y in ys for x in xs]

    # ------------------------------------------------------------ frame
    def infer_windows(self, frame: np.ndarray):
        """Roda os dois modelos em todas as janelas do quadro."""
        t0 = time.perf_counter()
        ox, oy = 0, 0
        img = frame
        if self.roi:
            ox, oy, rw, rh = map(int, self.roi)
            img = frame[oy:oy + rh, ox:ox + rw]
        h, w = img.shape[:2]
        wins = self._windows(w, h)
        crops = [img[y:y + self.tile, x:x + self.tile] for x, y in wins]
        t1 = time.perf_counter()
        probs = self.classifier.predict(crops)
        t2 = time.perf_counter()
        scores = self.padim.score(crops) / self.padim.threshold     # 1.0 = no limiar
        t3 = time.perf_counter()
        self.last_ms = {"crops": len(crops), "classifier_ms": (t2 - t1) * 1000, "padim_ms": (t3 - t2) * 1000}
        return {"wins": wins, "probs": probs, "scores": scores, "offset": (ox, oy), "size": (w, h), "t0": t0}

    def detect(self, frame: np.ndarray) -> list[Detection]:
        out = self.aggregate(self.infer_windows(frame))
        return out

    def aggregate(self, r: dict, mode: str | None = None, foreign_conf: float | None = None,
                  normal_conf: float | None = None) -> list[Detection]:
        """Transforma as saídas por janela em detecções.

        grid   (padrão) - o quadro é dividido em células do tamanho do passo; cada
                 célula recebe a MÉDIA das probabilidades de todas as janelas que a
                 cobrem. Um corpo estranho real fica alto em todas; erros isolados
                 de uma janela se diluem. Células alarmadas vizinhas viram uma caixa.
        window - cada janela decide sozinha (mais sensível, mais falso alarme)."""
        mode = mode or self.mode
        fc = self.dec.foreign_conf if foreign_conf is None else foreign_conf
        nc = self.dec.normal_conf if normal_conf is None else normal_conf
        wins, probs, scores = r["wins"], r["probs"], r["scores"]
        ox, oy = r["offset"]
        names = self.classifier.names
        if mode == "window":
            decs = decide(probs, scores, names, self.normal, 1.0, fc, nc)
            dets = [Detection((x + ox, y + oy, self.tile, self.tile), d.status, d.label, d.confidence, d.anomaly)
                    for (x, y), d in zip(wins, decs)]
            out = _merge([d for d in dets if d.status != OK]) + [d for d in dets if d.status == OK]
        else:
            out = self._grid(wins, probs, scores, r["size"], (ox, oy), fc, nc)
        if "t0" in r:
            self.last_ms["total_ms"] = (time.perf_counter() - r["t0"]) * 1000
        return out

    def _grid(self, wins, probs, scores, size, offset, fc, nc) -> list[Detection]:
        s, t = self.stride, self.tile
        w, h = size
        ox, oy = offset
        gw, gh = -(-w // s), -(-h // s)
        k = probs.shape[1]
        sp = np.zeros((gh, gw, k), np.float32)
        sa = np.zeros((gh, gw), np.float32)
        n = np.zeros((gh, gw), np.float32)
        for (x, y), p, a in zip(wins, probs, scores):
            c0, r0 = x // s, y // s
            c1, r1 = min(gw, -(-(x + t) // s)), min(gh, -(-(y + t) // s))
            sp[r0:r1, c0:c1] += p
            sa[r0:r1, c0:c1] += a
            n[r0:r1, c0:c1] += 1
        valid = n > 0
        n = np.maximum(n, 1)
        mp, ma = sp / n[..., None], sa / n
        flat = decide(mp.reshape(-1, k), ma.reshape(-1), self.classifier.names, self.normal, 1.0, fc, nc)
        status = np.array([d.status for d in flat]).reshape(gh, gw)
        status[~valid] = OK
        dets: list[Detection] = []
        for st in (FOREIGN, UNKNOWN):
            mask = (status == st).astype(np.uint8)
            if not mask.any():
                continue
            nlab, lab = cv2.connectedComponents(mask, connectivity=8)
            for i in range(1, nlab):
                rr, cc = np.where(lab == i)
                p = mp[rr, cc].mean(0)
                fidx = [j for j, nm in enumerate(self.classifier.names) if nm not in self.normal]
                top = max(fidx, key=lambda j: p[j]) if st == FOREIGN else int(p.argmax())
                x0, y0 = cc.min() * s, rr.min() * s
                x1, y1 = min(w, (cc.max() + 1) * s), min(h, (rr.max() + 1) * s)
                dets.append(Detection((int(x0 + ox), int(y0 + oy), int(x1 - x0), int(y1 - y0)), st,
                                      self.classifier.names[top] if st == FOREIGN else "Unknown",
                                      float(p[top]), float(ma[rr, cc].max())))
        # "Desconhecido" encostado num corpo estranho já identificado é a borda
        # do mesmo objeto: mostra só a caixa vermelha
        foreign = [d.box for d in dets if d.status == FOREIGN]
        return [d for d in dets if d.status == FOREIGN or not any(_touch(d.box, f) for f in foreign)]


def _touch(a, b) -> bool:
    """Caixas se sobrepõem ou encostam."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax <= bx + bw and bx <= ax + aw and ay <= by + bh and by <= ay + ah


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter + 1e-9)


def _merge(dets: list[Detection], iou_thr: float = 0.2) -> list[Detection]:
    """NMS: corpo estranho conhecido tem prioridade sobre desconhecido;
    dentro do mesmo nível vence a maior confiança/anomalia."""
    def key(d):
        return (d.status == FOREIGN, d.confidence if d.status == FOREIGN else d.anomaly)

    keep: list[Detection] = []
    for d in sorted(dets, key=key, reverse=True):
        if all(_iou(d.box, k.box) < iou_thr for k in keep):
            keep.append(d)
    return keep


# ------------------------------------------------------------ desenho
COLORS = {FOREIGN: (0, 0, 255), UNKNOWN: (0, 215, 255), OK: (0, 200, 0)}


def draw(frame: np.ndarray, dets: list[Detection], names: dict, stats: dict, show_unknown=True,
         show_normal=False, roi=None) -> np.ndarray:
    out = frame.copy()
    if roi:
        x, y, w, h = map(int, roi)
        cv2.rectangle(out, (x, y), (x + w, y + h), (200, 200, 200), 1)
    for d in dets:
        if d.status == OK and not show_normal:
            continue
        if d.status == UNKNOWN and not show_unknown:
            continue
        x, y, w, h = d.box
        c = COLORS[d.status]
        cv2.rectangle(out, (x, y), (x + w, y + h), c, 3 if d.status != OK else 1)
        if d.status != OK:
            txt = f"{names.get(d.label, d.label)} {d.confidence:.0%}" if d.status == FOREIGN \
                else f"{names.get('Unknown', 'Desconhecido')} a={d.anomaly:.1f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x, y - th - 8), (x + tw + 6, y), c, -1)
            cv2.putText(out, txt, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    # painel
    lines = [f"FPS: {stats.get('fps', 0):.1f}   inferencia: {stats.get('total_ms', 0):.0f} ms",
             f"Corpos estranhos (total): {stats.get('n_foreign', 0)}",
             f"Desconhecidos (total): {stats.get('n_unknown', 0)}",
             f"Neste quadro: {stats.get('frame_alarms', 0)}   camera: {stats.get('camera', '')}",
             "[q] sair  [p] pausa  [s] salvar quadro"]
    pw = 380
    cv2.rectangle(out, (0, 0), (pw, 22 * len(lines) + 10), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(out, t, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out
