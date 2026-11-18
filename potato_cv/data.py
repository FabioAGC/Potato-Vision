"""Preparação do dataset.

1. Junta todas as imagens da pasta de origem (a classe é o nome da pasta-mãe,
   ex.: origem/Stone/a.bmp ou origem/train/Stone/a.bmp).
2. Remove duplicatas - inclusive cópias giradas/espelhadas da mesma foto,
   que fariam o teste "colar" do treino.
3. Redivide em treino/validação/teste estratificado por classe.
4. No treino, completa as classes pequenas com rotações/espelhamentos.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .config import IMG_EXTS


# ---------------------------------------------------------------- IO
def imread(path: Path) -> np.ndarray | None:
    """cv2.imread que funciona com qualquer caminho do Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(Path(path).suffix or ".png", img)
    if ok:
        buf.tofile(str(path))


def dihedral(img: np.ndarray, k: int) -> np.ndarray:
    """k em 0..7: rotação de (k//2)*90 graus, espelhada se k for ímpar."""
    r = np.rot90(img, k // 2)
    return np.ascontiguousarray(r[:, ::-1] if k % 2 else r)


def _canonical_hash(img: np.ndarray) -> str:
    """Igual para a imagem e suas 8 rotações/espelhamentos."""
    return min(hashlib.md5(v.tobytes() + str(v.shape).encode()).hexdigest()
               for v in (dihedral(img, k) for k in range(8)))


# ---------------------------------------------------------------- scan
def scan_source(source: Path | list[Path]) -> tuple[dict[str, list[Path]], dict]:
    """Agrupa imagens únicas por classe. Aceita uma pasta ou uma lista de pastas
    (ex.: dataset original + dataset_roboflow). Pastas cujo nome começa com "_"
    (ex.: _frames) são ignoradas."""
    sources = [Path(s) for s in (source if isinstance(source, (list, tuple)) else [source])]
    files = sorted(p for src in sources for p in src.rglob("*")
                   if p.suffix.lower() in IMG_EXTS and not any(x.startswith("_") for x in p.relative_to(src).parts))
    if not files:
        raise SystemExit(f"Nenhuma imagem encontrada em {source}")
    labels: dict[str, set[str]] = defaultdict(set)
    first: dict[str, Path] = {}
    unreadable = 0
    for p in files:
        img = imread(p)
        if img is None:
            unreadable += 1
            continue
        h = _canonical_hash(img)
        labels[h].add(p.parent.name)
        first.setdefault(h, p)

    # mesma imagem com rótulos diferentes -> descartada (rótulo ambíguo)
    conflicts = sum(len(c) > 1 for c in labels.values())
    per_class: dict[str, list[Path]] = defaultdict(list)
    for h, cls in labels.items():
        if len(cls) == 1:
            per_class[next(iter(cls))].append(first[h])
    per_class = {c: sorted(v) for c, v in sorted(per_class.items())}
    info = {
        "source": [str(s) for s in sources] if len(sources) > 1 else str(sources[0]),
        "files_per_source": {str(s): sum(1 for p in files if src_of(p, s)) for s in sources},
        "files_found": len(files),
        "unreadable": unreadable,
        "unique_images": len(labels),
        "duplicates_removed": len(files) - unreadable - len(labels),
        "label_conflicts_removed": conflicts,
        "unique_per_class": {c: len(v) for c, v in per_class.items()},
    }
    return per_class, info


def src_of(p: Path, src: Path) -> bool:
    try:
        p.relative_to(src)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------- write
def write_split(groups: dict[str, list[Path]], out: Path, balance: bool) -> dict[str, int]:
    """Grava as imagens em out/<classe>/. Com balance=True completa as classes
    menores com rotações/espelhamentos (até 8x) rumo ao tamanho da maior."""
    target = max((len(v) for v in groups.values()), default=0)
    counts = Counter()
    for cls, paths in groups.items():
        d = out / cls
        d.mkdir(parents=True, exist_ok=True)
        imgs = [imread(p) for p in paths]
        n = len(imgs)
        if not n:
            continue
        total = min(8 * n, max(n, target)) if balance else n
        for j in range(total):
            k, i = divmod(j, n)
            imwrite(d / f"{cls}_{i:05d}_v{k}.png", dihedral(imgs[i], k))
        counts[cls] = total
    return dict(counts)


def split_groups(per_class: dict[str, list[Path]], split, seed: int) -> dict[str, dict[str, list[Path]]]:
    rng = random.Random(seed)
    parts: dict[str, dict[str, list[Path]]] = {"train": {}, "val": {}, "test": {}}
    for cls, paths in per_class.items():
        paths = paths[:]
        rng.shuffle(paths)
        n = len(paths)
        n_val, n_test = max(1, round(n * split[1])), max(1, round(n * split[2]))
        parts["test"][cls] = paths[:n_test]
        parts["val"][cls] = paths[n_test:n_test + n_val]
        parts["train"][cls] = paths[n_test + n_val:]
    return parts


def kfold(per_class: dict[str, list[Path]], k: int, seed: int) -> list[dict[str, list[Path]]]:
    """Divide cada classe em k partes (estratificado)."""
    rng = random.Random(seed)
    folds: list[dict[str, list[Path]]] = [defaultdict(list) for _ in range(k)]
    for cls, paths in per_class.items():
        paths = paths[:]
        rng.shuffle(paths)
        for i, p in enumerate(paths):
            folds[i % k][cls].append(p)
    return [dict(f) for f in folds]


def merge_groups(*gs: dict[str, list[Path]]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = defaultdict(list)
    for g in gs:
        for c, v in g.items():
            out[c] += v
    return dict(out)


def make_windows(groups: dict[str, list[Path]], out: Path, n: int, normal: set[str], seed: int,
                 foreign_cell_rate: float = 0.25) -> dict[str, int]:
    """Janelas desalinhadas: monta mosaicos 3x3 de recortes e corta uma janela
    do tamanho de um recorte numa posição aleatória - como acontece com a
    janela deslizante sobre o quadro real da câmera.

    Rótulo: corpo estranho se ele cobre >= 35% da janela; normal se os corpos
    estranhos cobrem <= 10% (classe normal com maior área); senão descarta."""
    rng = np.random.default_rng(seed)
    imgs = {c: [imread(p) for p in ps] for c, ps in groups.items() if ps}
    norm_c = [c for c in imgs if c in normal]
    for_c = [c for c in imgs if c not in normal]
    if not norm_c:
        return {}
    s = next(iter(imgs.values()))[0].shape[0]
    counts = Counter()
    made = tries = 0
    while made < n and tries < n * 5:
        tries += 1
        cells, mosaic = [], np.zeros((3 * s, 3 * s, 3), np.uint8)
        for r in range(3):
            for q in range(3):
                pool = for_c if (for_c and rng.random() < foreign_cell_rate) else norm_c
                c = pool[rng.integers(len(pool))]
                im = cv2.resize(dihedral(imgs[c][rng.integers(len(imgs[c]))], int(rng.integers(8))), (s, s))
                mosaic[r * s:(r + 1) * s, q * s:(q + 1) * s] = im
                cells.append((r, q, c))
        x, y = rng.integers(s // 2, 3 * s // 2 + 1, size=2)   # janela sempre toca a célula central
        cover = Counter()
        for r, q, c in cells:
            ix = max(0, min(x + s, (q + 1) * s) - max(x, q * s))
            iy = max(0, min(y + s, (r + 1) * s) - max(y, r * s))
            cover[c] += ix * iy / (s * s)
        f_cov = {c: v for c, v in cover.items() if c not in normal}
        if f_cov and max(f_cov.values()) >= 0.35:
            label = max(f_cov, key=f_cov.get)
        elif sum(f_cov.values()) <= 0.10:
            label = max((c for c in cover if c in normal), key=cover.get)
        else:
            continue
        d = out / label
        d.mkdir(parents=True, exist_ok=True)
        imwrite(d / f"win_{made:05d}.png", mosaic[y:y + s, x:x + s])
        counts[label] += 1
        made += 1
    return dict(counts)


def write_parts(parts: dict[str, dict[str, list[Path]]], out: Path, normal: set[str] | None = None,
                windows: dict[str, int] | None = None, seed: int = 0) -> dict:
    if out.exists():
        shutil.rmtree(out)
    res = {}
    for i, (s, g) in enumerate(parts.items()):
        res[s] = write_split(g, out / s, balance=(s == "train"))
        if normal and windows and windows.get(s):
            res[s + "_janelas"] = make_windows(g, out / s, windows[s], normal, seed + i)
    return res


def prepare_dataset(source: Path | list[Path], out: Path, split=(0.7, 0.15, 0.15), seed: int = 42,
                    normal: set[str] | None = None, windows: dict[str, int] | None = None) -> dict:
    per_class, info = scan_source(source)
    parts = split_groups(per_class, split, seed)
    written = write_parts(parts, out, normal, windows, seed)
    report = {
        **info,
        "splits_unique": {s: {c: len(v) for c, v in g.items()} for s, g in parts.items()},
        "splits_written": written,
        "classes": sorted(per_class),
    }
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ---------------------------------------------------------------- read
def list_split(prepared: Path, split: str, originals_only: bool = False, kind: str | None = None
               ) -> list[tuple[Path, str]]:
    """(caminho, classe) de um split.
    kind: None/"all" = tudo | "orig" = recortes originais (sem rotações) |
          "win" = janelas desalinhadas | "orig+win"."""
    if originals_only and kind is None:
        kind = "orig"
    pats = {"orig": ["*_v0.png"], "win": ["win_*.png"], "orig+win": ["*_v0.png", "win_*.png"]}.get(kind or "all", ["*.png"])
    items = []
    for cls_dir in sorted((Path(prepared) / split).iterdir()):
        if cls_dir.is_dir():
            for pat in pats:
                items += [(p, cls_dir.name) for p in sorted(cls_dir.glob(pat))]
    return items


def load_images(paths) -> list[np.ndarray]:
    return [imread(p) for p in paths]
