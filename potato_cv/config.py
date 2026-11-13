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
