"""Classificador YOLO (Ultralytics, modo classificação) para os recortes."""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import ROOT


def train_classifier(data_dir: Path, out_weights: Path, model: str = "yolo11n-cls.pt", imgsz: int = 64,
                     epochs: int = 60, batch: int = 64, patience: int = 15, name: str = "classifier",
                     seed: int = 0, verbose: bool = False, device: str = "cpu", workers: int = 0,
                     auto_augment: str | None = None, extra: dict | None = None) -> Path:
    """Treina em data_dir/{train,val}/<classe>/ e copia o melhor peso para out_weights.
    `device`: "cpu" ou "cuda"/"0" (GPU). `extra`: argumentos adicionais do Ultralytics
    (ex.: {"lr0": 0.001, "optimizer": "AdamW"}), sobrescrevem os padrões abaixo."""
    from ultralytics import YOLO

    if device.startswith("cuda"):  # Ultralytics espera "0", "0,1"...
        device = device.split(":", 1)[1] if ":" in device else "0"
    args = dict(data=str(data_dir), imgsz=imgsz, epochs=epochs, batch=batch, patience=patience,
                device=device, workers=workers, cache="ram", project=str(ROOT / "runs"), name=name, exist_ok=True,
                seed=seed, deterministic=True, plots=False, verbose=verbose,
                # RandAugment desligado por padrão: deixava o treino em CPU 4x mais lento sem ganho.
                # Em GPU vale testar auto_augment: randaugment (config.kaggle.yaml).
                auto_augment=auto_augment, fliplr=0.5, flipud=0.5, hsv_h=0.01, hsv_s=0.3, hsv_v=0.3, erasing=0.2)
    args.update(extra or {})
    yolo = YOLO(model)
    yolo.train(**args)
    best = Path(yolo.trainer.best)
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out_weights)
    return out_weights


class Classifier:
    """Inferência em lote sobre crops BGR, chamando a rede diretamente
    (o `predict` do Ultralytics processa imagem a imagem e é ~5x mais lento).
    Aceita .pt (PyTorch) ou .onnx (ONNX Runtime)."""

    def __init__(self, weights: Path, imgsz: int = 64, device: str = "cpu"):
        from ultralytics import YOLO

        self.weights = Path(weights)
        self.device = device
        self.imgsz = imgsz
        pt = self.weights.with_suffix(".pt")
        yolo = YOLO(str(pt), task="classify")
        self.names: list[str] = [yolo.names[i] for i in sorted(yolo.names)]
        self.session = None
        if self.weights.suffix == ".onnx":
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(self.weights), providers=ort.get_available_providers())
            self.input_name = self.session.get_inputs()[0].name
            self.net = None
        else:
            self.net = yolo.model.float().eval().to(device)
            for p in self.net.parameters():
                p.requires_grad_(False)
        self._yolo = yolo

    def _prep(self, crops: list[np.ndarray]) -> np.ndarray:
        # igual ao treino do Ultralytics: RGB, redimensiona p/ imgsz, 0..1
        arr = np.stack([cv2.resize(c, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
                        for c in crops]).astype(np.float32) / 255.0
        return np.ascontiguousarray(arr.transpose(0, 3, 1, 2))

    @torch.no_grad()
    def predict(self, crops: list[np.ndarray], batch: int = 256) -> np.ndarray:
        """Probabilidades (N, C) na ordem de self.names."""
        if not crops:
            return np.zeros((0, len(self.names)), dtype=np.float32)
        out = []
        for i in range(0, len(crops), batch):
            x = self._prep(crops[i:i + batch])
            if self.session is not None:
                y = self.session.run(None, {self.input_name: x})[0]
            else:
                y = self.net(torch.from_numpy(x).to(self.device))
                y = (y[0] if isinstance(y, (list, tuple)) else y).cpu().numpy()
            out.append(y)
        return np.concatenate(out).astype(np.float32)

    def export_onnx(self) -> Path:
        return Path(self._yolo.export(format="onnx", imgsz=self.imgsz, dynamic=True, simplify=False))
