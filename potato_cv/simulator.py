"""Esteira simulada: monta quadros com recortes reais do dataset (do split de
teste, nunca vistos no treino) rolando para baixo. Serve para testar o
sistema completo enquanto a câmera não chega - e como sabe onde cada objeto
está, também permite medir o acerto quadro a quadro."""
from __future__ import annotations

import random
import time
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera
from .data import dihedral, imread


class ConveyorSimulator(Camera):
    name = "simulator"

    def __init__(self, folder: Path, width=960, height=640, object_px=100, foreign_rate=0.06,
                 speed_px=4, normal_classes=frozenset({"Good", "Damaged"}), fps=30, seed=0):
        self.w, self.h, self.s = int(width), int(height), int(object_px)
        self.speed, self.fps = int(speed_px), fps
        self.foreign_rate = foreign_rate
        self.rng = random.Random(seed)
        # colunas cobrem a largura toda (a última pode ficar cortada): na
        # instalação real a ROI deve enquadrar só a esteira
        self.cols = max(1, -(-self.w // self.s))
        self.margin = 0
        self.normal, self.foreign = [], []
        for cls_dir in sorted(Path(folder).iterdir()):
            if not cls_dir.is_dir():
                continue
            files = sorted(cls_dir.glob("*_v0.png")) or sorted(cls_dir.glob("*.png"))
            imgs = [(imread(p), cls_dir.name) for p in files]
            (self.normal if cls_dir.name in normal_classes else self.foreign).extend(imgs)
        if not self.normal:
            raise FileNotFoundError(f"Simulador: sem imagens em {folder}. Rode o treino primeiro (prepara os dados).")
        self.rows: list[tuple[float, list[tuple[np.ndarray, str]]]] = []
        y = self.h
        while y > -self.s:
            y -= self.s
            self.rows.append((y, self._new_row()))
        self._last = 0.0

    def _new_row(self):
        self._oid = getattr(self, "_oid", 0)
        row = []
        for _ in range(self.cols):
            pool = self.foreign if (self.foreign and self.rng.random() < self.foreign_rate) else self.normal
            img, cls = self.rng.choice(pool)
            img = cv2.resize(dihedral(img, self.rng.randrange(8)), (self.s, self.s), interpolation=cv2.INTER_LINEAR)
            row.append((img, cls, self._oid))
            self._oid += 1
        return row

    def _step(self):
        self.rows = [(y + self.speed, r) for y, r in self.rows if y + self.speed < self.h]
        top = min(y for y, _ in self.rows)
        while top > -self.s:
            top -= self.s
            self.rows.append((top, self._new_row()))

    def ground_truth(self) -> list[tuple[tuple[int, int, int, int], str, int]]:
        """[(x, y, w, h), classe, id] de cada objeto visível no quadro atual."""
        gt = []
        for y, row in self.rows:
            for j, (_, cls, oid) in enumerate(row):
                y0, y1 = max(0, int(y)), min(self.h, int(y) + self.s)
                x0, x1 = j * self.s, min(self.w, (j + 1) * self.s)
                if y1 - y0 > self.s // 2 and x1 - x0 > self.s // 2:
                    gt.append(((x0, y0, x1 - x0, y1 - y0), cls, oid))
        return gt

    def read(self):
        # respeita o FPS configurado, como uma câmera real
        wait = 1.0 / self.fps - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        self._step()
        frame = np.full((self.h, self.w, 3), (95, 85, 80), np.uint8)  # cor da esteira
        for y, row in self.rows:
            y = int(y)
            y0, y1 = max(0, y), min(self.h, y + self.s)
            if y1 <= y0:
                continue
            for j, (img, _, _) in enumerate(row):
                x = j * self.s
                x1 = min(self.w, x + self.s)
                frame[y0:y1, x:x1] = img[y0 - y:y1 - y, :x1 - x]
        return True, frame
