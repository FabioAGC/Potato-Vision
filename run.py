"""Detecção ao vivo com tela.

    python run.py                          # usa camera.type do config.yaml
    python run.py --camera usb             # força um tipo de câmera
    python run.py --camera rtsp --url rtsp://...
    python run.py --headless --frames 300  # sem janela (teste/benchmark)

Teclas: q = sair, p = pausar, s = salvar quadro em data/snapshots/
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from potato_cv.camera import ThreadedCamera, create_camera
from potato_cv.config import ROOT, load_config
from potato_cv.data import imwrite
from potato_cv.decision import FOREIGN, OK, UNKNOWN
from potato_cv.detector import FrameDetector, draw

log = logging.getLogger("run")


class Tracker:
    """Rastreador por proximidade com confirmação temporal.

    Um alarme só é confirmado (desenhado e contado) depois de aparecer em
    `confirm` quadros processados seguidos perto do mesmo lugar - elimina
    alarmes que piscam em um único quadro. Cada objeto é contado uma vez."""

    def __init__(self, max_dist: float, confirm: int = 3, max_missing: int = 1, keep: int = 6):
        # max_missing: falhas toleradas na sequência de confirmação
        # keep: quadros que o objeto fica na memória (evita contar 2x o mesmo)
        self.max_dist, self.confirm, self.max_missing, self.keep = max_dist, confirm, max_missing, keep
        self.tracks: dict[int, dict] = {}
        self.next_id = 0
        self.counts = {FOREIGN: 0, UNKNOWN: 0}

    def update(self, dets):
        """Retorna as detecções confirmadas (+ todas as OK, sem alteração)."""
        for t in self.tracks.values():
            t["missing"] += 1
            if t["missing"] > self.max_missing and t["counted"] is None:
                t["hits"] = 0          # sequência quebrada: recomeça a confirmação
        confirmed = []
        used = set()
        for d in sorted((d for d in dets if d.status != OK), key=lambda d: d.status != FOREIGN):
            x, y, w, h = d.box
            c = np.array([x + w / 2, y + h / 2])
            best, best_d = None, self.max_dist
            for tid, t in self.tracks.items():
                if tid in used:
                    continue
                dist = np.linalg.norm(t["c"] - c)
                if dist < best_d:
                    best, best_d = tid, dist
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.tracks[best] = {"c": c, "missing": 0, "hits": 0, "status": d.status, "counted": None}
            t = self.tracks[best]
            used.add(best)
            t["c"], t["missing"] = c, 0
            t["hits"] += 1
            if t["counted"] is not None:   # já confirmado: continua desenhando
                t["hits"] = max(t["hits"], self.confirm)
            if d.status == FOREIGN:
                t["status"] = FOREIGN   # conhecido prevalece sobre desconhecido
            if t["hits"] >= self.confirm:
                if t["counted"] is None:
                    self.counts[t["status"]] += 1
                    t["counted"] = t["status"]
                elif t["counted"] == UNKNOWN and t["status"] == FOREIGN:
                    self.counts[UNKNOWN] -= 1
                    self.counts[FOREIGN] += 1
                    t["counted"] = FOREIGN
                confirmed.append(d)
        self.tracks = {k: t for k, t in self.tracks.items() if t["missing"] <= self.keep}
        return confirmed + [d for d in dets if d.status == OK]


def main():
    ap = argparse.ArgumentParser(description="Detecção de corpos estranhos ao vivo.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--camera", default=None, help="simulator | usb | rtsp | genicam | video | images")
    ap.add_argument("--url", default=None, help="URL RTSP (com --camera rtsp)")
    ap.add_argument("--index", type=int, default=None, help="índice da câmera USB")
    ap.add_argument("--video", default=None, help="arquivo de vídeo (com --camera video)")
    ap.add_argument("--headless", action="store_true", help="sem janela")
    ap.add_argument("--frames", type=int, default=0, help="para após N quadros (0 = infinito)")
    ap.add_argument("--sim-speed", type=int, default=None, help="velocidade da esteira simulada (px/quadro)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(a.config)
    if a.camera:
        cfg["camera"]["type"] = a.camera
    if a.url:
        cfg["camera"]["rtsp"]["url"] = a.url
    if a.index is not None:
        cfg["camera"]["usb"]["index"] = a.index
    if a.video:
        cfg["camera"]["video"]["path"] = a.video
    if a.sim_speed:
        cfg["camera"]["simulator"]["speed_px"] = a.sim_speed
        cfg["camera"]["fps"] = 1000  # teste acelerado: não espera o relógio

    det = FrameDetector(cfg)
    cam = create_camera(cfg)
    names = dict(cfg.display_names)
    tracker = Tracker(max_dist=det.tile * 1.0, confirm=int(cfg.inference.get("confirm_frames", 3)),
                      max_missing=int(cfg.inference.get("max_missing", 1)))
    disp = cfg.display
    win = disp.window
    sim_eval = {"gt_foreign": set(), "gt_normal": set(), "hit": set(), "false": set()}

    log.info("Câmera: %s | tile=%dpx stride=%dpx | Ctrl+C ou 'q' para sair", cam.name, det.tile, det.stride)
    cam.open()
    try:
        n, seq, fps, t_last = 0, 0, 0.0, time.time()
        dets, paused, frame = [], False, None
        every = max(1, int(cfg.inference.every_n_frames))
        while True:
            if not paused:
                if isinstance(cam, ThreadedCamera):
                    seq, frame = cam.read_new(seq, timeout=2.0)
                    ok = frame is not None
                else:
                    ok, frame = cam.read()
                if not ok:
                    if not getattr(cam, "connected", True):
                        time.sleep(0.2)
                        continue
                    if cfg.camera.type in ("video", "images"):
                        break
                    continue
                if n % every == 0:
                    dets = tracker.update(det.detect(frame))
                    if hasattr(cam, "ground_truth"):
                        _sim_match(cam, dets, cfg, sim_eval)
                n += 1
                now = time.time()
                fps = 0.9 * fps + 0.1 / max(1e-6, now - t_last) if fps else 1 / max(1e-6, now - t_last)
                t_last = now

            alarms = [d for d in dets if d.status != OK]
            stats = {"fps": fps, "total_ms": det.last_ms.get("total_ms", 0), "camera": cam.name,
                     "n_foreign": tracker.counts[FOREIGN], "n_unknown": tracker.counts[UNKNOWN],
                     "frame_alarms": len(alarms)}
            if a.headless:
                if n % 30 == 0:
                    log.info("quadro %d | %.1f FPS | %.0f ms | corpos estranhos %d | desconhecidos %d",
                             n, fps, stats["total_ms"], stats["n_foreign"], stats["n_unknown"])
            else:
                vis = draw(frame, dets, names, stats, disp.show_unknown, disp.show_normal, cfg.inference.get("roi"))
                cv2.imshow(win, vis)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if k == ord("p"):
                    paused = not paused
                if k == ord("s"):
                    p = ROOT / "data" / "snapshots" / time.strftime("%Y%m%d_%H%M%S.png")
                    p.parent.mkdir(parents=True, exist_ok=True)
                    imwrite(p, frame)
                    log.info("Quadro salvo: %s", p)
            if a.frames and n >= a.frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        cv2.destroyAllWindows()
    log.info("Fim. Corpos estranhos: %d | Desconhecidos: %d | quadros: %d",
             tracker.counts[FOREIGN], tracker.counts[UNKNOWN], n)
    if sim_eval["gt_foreign"]:
        g, h = len(sim_eval["gt_foreign"]), len(sim_eval["hit"] & sim_eval["gt_foreign"])
        nn, fa = len(sim_eval["gt_normal"]), len(sim_eval["false"] & sim_eval["gt_normal"])
        log.info("Simulador: %d de %d corpos estranhos alarmados (%.0f%%) | %d de %d batatas com falso alarme (%.1f%%)",
                 h, g, 100 * h / g, fa, nn, 100 * fa / max(1, nn))


def _sim_match(cam, dets, cfg, ev):
    """No simulador (que sabe onde está cada objeto), mede por OBJETO que passa:
    quantos corpos estranhos foram alarmados e quantas batatas geraram falso alarme."""
    normal = set(cfg.normal_classes)
    alarms = [d for d in dets if d.status != OK]
    gt = cam.ground_truth()
    for (x, y, w, h), cls, oid in gt:
        if h >= w:  # objeto inteiro no quadro
            ev["gt_foreign" if cls not in normal else "gt_normal"].add(oid)
    for d in alarms:
        dx, dy, dw, dh = d.box
        cx, cy = dx + dw / 2, dy + dh / 2
        under = [(cls, oid) for (x, y, w, h), cls, oid in gt if x <= cx < x + w and y <= cy < y + h]
        if not under:
            continue
        # acerto se algum corpo estranho toca a janela alarmada
        touched = [oid for (x, y, w, h), cls, oid in gt if cls not in normal
                   and x < dx + dw and dx < x + w and y < dy + dh and dy < y + h]
        if touched:
            ev["hit"].update(touched)
        else:
            ev["false"].add(under[0][1])


if __name__ == "__main__":
    main()
