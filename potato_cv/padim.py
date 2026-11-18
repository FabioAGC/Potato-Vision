"""PaDiM - Patch Distribution Modeling (Defard et al., 2020).

Aprende como é uma batata NORMAL a partir de features de uma ResNet
pré-treinada (ImageNet). Para cada posição da imagem guarda uma gaussiana
multivariada; na inferência, a distância de Mahalanobis de cada patch até
essa gaussiana vira um mapa de anomalia. Qualquer coisa diferente de batata
(pedra, galho, plástico, metal...) tende a ter distância alta.

Treina só com imagens normais, por isso detecta objetos nunca vistos.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FeatureExtractor(torch.nn.Module):
    """ResNet -> layer1, layer2, layer3 reamostradas e concatenadas."""

    def __init__(self, backbone: str = "resnet18", idx: torch.Tensor | None = None):
        super().__init__()
        weights = {"resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
                   "wide_resnet50_2": torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1}[backbone]
        net = getattr(torchvision.models, backbone)(weights=weights)
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        self.register_buffer("idx", idx if idx is not None else torch.arange(1))
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: float RGB 0..1, NCHW
        x = (x - _MEAN.to(x.device)) / _STD.to(x.device)
        f1 = self.layer1(self.stem(x))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        size = f1.shape[-2:]
        emb = torch.cat([f1,
                         F.interpolate(f2, size=size, mode="nearest"),
                         F.interpolate(f3, size=size, mode="nearest")], dim=1)
        return emb.index_select(1, self.idx)


def to_tensor(crops: list[np.ndarray], size: int) -> torch.Tensor:
    """Lista de crops BGR uint8 -> tensor RGB float NCHW."""
    import cv2
    arr = np.stack([cv2.resize(c, (size, size), interpolation=cv2.INTER_LINEAR)[:, :, ::-1] for c in crops])
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2).float() / 255.0


class PaDiM:
    def __init__(self, backbone="resnet18", input_size=48, feature_dims=200, score_mode="mean", seed=0,
                 device: str = "cpu"):
        self.device = device
        self.backbone, self.input_size, self.feature_dims = backbone, input_size, feature_dims
        self.score_mode = score_mode
        full = {"resnet18": 448, "wide_resnet50_2": 1792}[backbone]
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(full, generator=g)[:feature_dims].sort().values
        self.extractor = FeatureExtractor(backbone, idx).to(device)
        self.mean = self.inv_cov = None
        self.threshold: float | None = None
        self.score_stats: dict = {}
        self.onnx_session = None

    # ------------------------------------------------------------ features
    @torch.no_grad()
    def _embed(self, crops: list[np.ndarray], batch: int = 64) -> torch.Tensor:
        out = []
        for i in range(0, len(crops), batch):
            x = to_tensor(crops[i:i + batch], self.input_size)
            if self.onnx_session is not None:
                y = self.onnx_session.run(None, {"input": x.numpy()})[0]
                out.append(torch.from_numpy(y).to(self.device))
            else:
                out.append(self.extractor(x.to(self.device)))
        return torch.cat(out)

    # ------------------------------------------------------------ fit
    @torch.no_grad()
    def fit(self, crops: list[np.ndarray], batch: int = 64) -> None:
        """Gaussiana por posição, acumulada em lotes (memória constante)."""
        s = ss = None
        n = 0
        for i in range(0, len(crops), batch):
            emb = self._embed(crops[i:i + batch], batch).double()   # B, D, H, W
            b, d, h, w = emb.shape
            e = emb.reshape(b, d, h * w)
            s = e.sum(0) if s is None else s + e.sum(0)                         # D, HW
            o = torch.einsum("bdp,bep->pde", e, e)                               # HW, D, D
            ss = o if ss is None else ss + o
            n += b
        mean = s / n                                                              # D, HW
        mt = mean.T                                                               # HW, D
        cov = (ss - n * mt[:, :, None] * mt[:, None, :]) / max(1, n - 1)
        cov += 0.01 * torch.eye(d, dtype=cov.dtype, device=cov.device)
        self.mean = mean.float()
        # Inversa sempre na CPU: é rápida (HW matrizes D x D) e evita as rotinas de
        # álgebra linear da GPU, que faltam em algumas placas/builds do PyTorch
        # ("no kernel image is available" na P100 do Kaggle).
        # Cholesky (cov é simétrica positiva-definida) com 1 thread: o inv
        # multi-thread travava no Windows depois do treino do YOLO no mesmo processo.
        prev = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            cov = cov.cpu()
            self.inv_cov = torch.cholesky_inverse(torch.linalg.cholesky(cov)).float().to(self.device)
        finally:
            torch.set_num_threads(prev)
        self.map_hw = (h, w)

    # ------------------------------------------------------------ score
    @torch.no_grad()
    def anomaly_maps(self, crops: list[np.ndarray], batch: int = 64) -> torch.Tensor:
        """Mapa de anomalia (N, S, S) na resolução input_size."""
        maps = []
        mean, inv_cov = self.mean.to(self.device), self.inv_cov.to(self.device)
        for i in range(0, len(crops), batch):
            emb = self._embed(crops[i:i + batch], batch)
            n, d, h, w = emb.shape
            delta = (emb.reshape(n, d, h * w) - mean).permute(2, 0, 1)       # HW, N, D
            m = (torch.bmm(delta, inv_cov) * delta).sum(-1)               # HW, N
            m = m.clamp_min(0).sqrt().T.reshape(n, 1, h, w)
            m = F.interpolate(m, size=self.input_size, mode="bilinear", align_corners=False)
            maps.append(_gaussian_blur(m, sigma=4).squeeze(1).cpu())
        return torch.cat(maps)

    def score(self, crops: list[np.ndarray]) -> np.ndarray:
        """Nota de anomalia por crop: média (padrão) ou máximo do mapa.
        A média funciona melhor aqui porque o objeto ocupa quase todo o crop."""
        if not crops:
            return np.zeros(0, dtype=np.float32)
        maps = self.anomaly_maps(crops)
        agg = maps.mean(dim=(1, 2)) if self.score_mode == "mean" else maps.amax(dim=(1, 2))
        return agg.numpy()

    # ------------------------------------------------------------ calibração
    def calibrate(self, normal_scores: np.ndarray, target_fpr: float) -> float:
        self.threshold = float(np.quantile(normal_scores, 1 - target_fpr))
        self.score_stats = {"normal_mean": float(normal_scores.mean()),
                            "normal_std": float(normal_scores.std()),
                            "target_fpr": target_fpr}
        return self.threshold

    # ------------------------------------------------------------ IO
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"backbone": self.backbone, "input_size": self.input_size,
                    "feature_dims": self.feature_dims, "score_mode": self.score_mode,
                    "idx": self.extractor.idx.cpu(), "mean": self.mean.cpu(), "inv_cov": self.inv_cov.cpu(),
                    "map_hw": self.map_hw,
                    "threshold": self.threshold, "score_stats": self.score_stats}, path)

    @classmethod
    def load(cls, path: Path, engine: str = "torch") -> "PaDiM":
        s = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls.__new__(cls)
        obj.device = "cpu"
        obj.backbone, obj.input_size, obj.feature_dims = s["backbone"], s["input_size"], s["feature_dims"]
        obj.score_mode = s.get("score_mode", "mean")
        obj.extractor = FeatureExtractor(obj.backbone, s["idx"])
        obj.mean, obj.inv_cov, obj.map_hw = s["mean"], s["inv_cov"], s["map_hw"]
        obj.threshold, obj.score_stats = s["threshold"], s["score_stats"]
        obj.onnx_session = None
        onnx_path = Path(path).with_suffix(".onnx")
        if engine == "onnx" and onnx_path.exists():
            import onnxruntime as ort
            obj.onnx_session = ort.InferenceSession(str(onnx_path), providers=ort.get_available_providers())
        return obj

    def export_onnx(self, path: Path) -> Path:
        """Exporta o extrator de features (a parte pesada) para ONNX.
        As estatísticas gaussianas ficam no .pt ao lado."""
        onnx_path = Path(path).with_suffix(".onnx")
        dummy = torch.zeros(1, 3, self.input_size, self.input_size)
        torch.onnx.export(self.extractor, dummy, str(onnx_path), input_names=["input"],
                          output_names=["features"], dynamic_axes={"input": {0: "n"}, "features": {0: "n"}},
                          opset_version=17, dynamo=False)
        return onnx_path


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = int(4 * sigma + 1) | 1
    t = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
    g = torch.exp(-t ** 2 / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, 1, k)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.transpose(2, 3))
