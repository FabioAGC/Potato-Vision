"""Leitura do config.yaml com caminhos resolvidos a partir da raiz do projeto."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class Config(dict):
    """dict com acesso por atributo (cfg.camera.type)."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(value) if isinstance(value, dict) else value


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))


def resolve(p: str | Path) -> Path:
    """Caminho relativo -> relativo à raiz do projeto."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def resolve_device(device: str | None = "auto") -> str:
    """'auto' -> 'cuda' se houver GPU, senão 'cpu'. Aceita também 'cuda', 'cuda:0', 'cpu'."""
    import torch
    device = str(device or "auto").lower()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        check_gpu(torch, int(device.split(":")[1]) if ":" in device else 0)
    return device


def check_gpu(torch, index: int = 0) -> None:
    """Falha com mensagem clara se a GPU não roda um teste simples com este PyTorch
    ('no kernel image is available' = PyTorch sem código para esta placa)."""
    if not torch.cuda.is_available():
        raise SystemExit("device=cuda, mas nenhuma GPU foi encontrada.")
    major, minor = torch.cuda.get_device_capability(index)
    name = torch.cuda.get_device_name(index)
    try:  # teste rápido
        x = torch.rand(64, 64, device=f"cuda:{index}")
        (x @ x.T).relu().sum().item()
    except Exception as e:
        raise SystemExit(f"A GPU {name} (sm_{major}{minor}) falhou num teste simples: {e}\n"
                         f"PyTorch {torch.__version__} compilado para: {', '.join(torch.cuda.get_arch_list())}. "
                         "Troque a GPU ou rode com --device cpu.")
