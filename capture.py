"""Captura de imagens da câmera para criar/ampliar o dataset.

    python capture.py --every 1.0                 # salva 1 quadro por segundo
    python capture.py --camera usb --every 0.5
    python capture.py --crops --label Good        # recorta janelas (tile_px) e salva
                                                  # direto em data/captures/Good/

Use com a linha rodando SÓ com batata boa para retreinar o PaDiM com
imagens reais da câmera (o passo mais importante na instalação).
"""
from __future__ import annotations

import argparse
import logging
import time

import cv2

from potato_cv.camera import ThreadedCamera, create_camera
from potato_cv.config import ROOT, load_config, resolve
from potato_cv.data import imwrite

log = logging.getLogger("capture")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--out", default="data/captures")
    ap.add_argument("--every", type=float, default=1.0, help="segundos entre capturas")
    ap.add_argument("--count", type=int, default=0, help="para após N capturas (0 = infinito)")
    ap.add_argument("--crops", action="store_true", help="salva recortes do tamanho tile_px em vez do quadro")
    ap.add_argument("--label", default="unlabeled", help="subpasta/classe dos recortes")
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(a.config)
    if a.camera:
        cfg["camera"]["type"] = a.camera
    out = resolve(a.out) / (a.label if a.crops else "frames")
    out.mkdir(parents=True, exist_ok=True)
    tile = int(cfg.inference.tile_px)

    cam = create_camera(cfg)
    cam.open()
    saved, last, seq = 0, 0.0, 0
    log.info("Salvando em %s  (q para sair)", out)
    try:
        while not a.count or saved < a.count:
            if isinstance(cam, ThreadedCamera):
                seq, frame = cam.read_new(seq)
                ok = frame is not None
            else:
                ok, frame = cam.read()
            if not ok:
                continue
            if time.time() - last >= a.every:
                last = time.time()
                stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(last * 1000) % 1000:03d}"
                if a.crops:
                    h, w = frame.shape[:2]
                    for y in range(0, h - tile + 1, tile):
                        for x in range(0, w - tile + 1, tile):
                            imwrite(out / f"{stamp}_{x}_{y}.png", frame[y:y + tile, x:x + tile])
                else:
                    imwrite(out / f"{stamp}.png", frame)
                saved += 1
                log.info("captura %d", saved)
            if not a.no_preview:
                cv2.imshow("Captura (q para sair)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
