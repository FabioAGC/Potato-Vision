"""Importa um dataset de DETECÇÃO do Roboflow (formato YOLOv8) para o formato
de recortes usado no treino (uma pasta por classe).

    python import_roboflow.py potato.v1i.yolov8.zip                # zip baixado do Roboflow
    python import_roboflow.py pasta_descompactada --out dataset_roboflow
    python import_roboflow.py x.zip --map potato=Good rock=Stone   # mapeamento manual

Cada caixa vira um recorte quadrado (com margem) em  <out>/<Classe>/*.png.
Os quadros inteiros e as caixas ficam em <out>/_frames/ (para avaliar o
detector no quadro inteiro); essa pasta começa com "_" e é ignorada no treino.

Cuidados:
* Cópias aumentadas pelo Roboflow (mesmo nome antes de ".rf.") são reduzidas
  a UMA por foto original - senão a mesma foto cai no treino e no teste.
* Classes sem correspondência (ver --map) são ignoradas com aviso.
"""
from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from potato_cv.config import IMG_EXTS, ROOT
from potato_cv.data import imread, imwrite

# nome da classe no Roboflow (minúsculo, contém) -> classe do projeto
AUTO_MAP = [
    (("damage", "rot", "defect", "bad", "diseas", "danific", "podre", "cut"), "Damaged"),
    (("stone", "rock", "clod", "pedra", "torr", "soil"), "Stone"),
    (("plant", "leaf", "leaves", "stem", "weed", "stalk", "haulm", "galho", "folha", "planta"), "Plant"),
    (("potato", "batata", "good", "tuber", "healthy"), "Good"),
]


def auto_class(name: str) -> str | None:
    n = name.lower()
    for keys, cls in AUTO_MAP:
        if any(k in n for k in keys):
            return cls
    return None


def original_stem(p: Path) -> str:
    """'IMG_12_jpg.rf.abc123.jpg' -> 'IMG_12_jpg' (Roboflow junta augmentations assim)."""
    return re.split(r"\.rf\.", p.name)[0] if ".rf." in p.name else p.stem


def square_crop(img: np.ndarray, cx: float, cy: float, w: float, h: float, margin: float) -> np.ndarray:
    """Recorte quadrado centrado na caixa; bordas fora da imagem são replicadas."""
    H, W = img.shape[:2]
    s = int(round(max(w, h) * (1 + margin)))
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    pad = max(0, -x0, -y0, x0 + s - W, y0 + s - H)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0 = x0 + pad, y0 + pad
    return img[y0:y0 + s, x0:x0 + s]


def find_root(base: Path) -> Path:
    ys = sorted(base.rglob("data.yaml"), key=lambda p: len(p.parts))
    if not ys:
        raise SystemExit(f"data.yaml não encontrado em {base} - exporte do Roboflow no formato YOLOv8.")
    return ys[0].parent


def label_for(img: Path) -> Path | None:
    """…/images/x.jpg -> …/labels/x.txt"""
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            lp = Path(*parts).with_suffix(".txt")
            return lp if lp.exists() else None
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="zip ou pasta exportada do Roboflow (YOLOv8)")
    ap.add_argument("--out", default="dataset_roboflow")
    ap.add_argument("--map", nargs="*", default=[], help="classe_roboflow=Classe (ex.: potato=Good)")
    ap.add_argument("--margin", type=float, default=0.10, help="margem do recorte (fração do lado da caixa)")
    ap.add_argument("--min-px", type=int, default=24, help="ignora caixas menores que isso (px)")
    ap.add_argument("--keep-augmented", action="store_true", help="não remove cópias aumentadas pelo Roboflow")
    a = ap.parse_args()

    src = Path(a.src)
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    tmp = None
    if src.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="rf_"))
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        src = tmp
    try:
        root = find_root(src)
        meta = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
        names = meta["names"]
        names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
        manual = dict(m.split("=", 1) for m in a.map)
        cmap = {i: manual.get(n, auto_class(n) if not manual else None) for i, n in enumerate(names)}
        print("Classes do Roboflow -> projeto:")
        for i, n in enumerate(names):
            print(f"  {n!r:>20} -> {cmap[i] or '(ignorada)'}")
        if not any(cmap.values()):
            raise SystemExit("Nenhuma classe mapeada. Use --map nome=Good|Damaged|Stone|Plant")

        imgs = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS and "images" in p.parts)
        groups: dict[str, list[Path]] = defaultdict(list)
        for p in imgs:
            groups[p.name if a.keep_augmented else original_stem(p)].append(p)
        chosen = [g[0] for g in groups.values()]   # 1 por foto original
        print(f"Imagens: {len(imgs)} -> {len(chosen)} fotos originais"
              + ("" if a.keep_augmented else " (cópias aumentadas removidas)"))

        if out.exists():
            shutil.rmtree(out)
        frames = out / "_frames"
        (frames / "images").mkdir(parents=True)
        (frames / "labels").mkdir(parents=True)
        counts, skipped_small, skipped_cls = Counter(), 0, Counter()
        for k, p in enumerate(chosen):
            lp = label_for(p)
            img = imread(p)
            if img is None or lp is None:
                continue
            H, W = img.shape[:2]
            stem = f"rf{k:05d}"
            imwrite(frames / "images" / f"{stem}.png", img)
            rows = []
            for j, line in enumerate(lp.read_text().split("\n")):
                v = line.split()
                if len(v) < 5:
                    continue
                ci = int(v[0])
                if len(v) == 5:
                    cx, cy, bw, bh = (float(x) for x in v[1:5])
                else:  # polígono (segmentação) -> caixa envolvente
                    xs, ys = [float(x) for x in v[1::2]], [float(y) for y in v[2::2]]
                    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
                cls = cmap.get(ci)
                rows.append(f"{cls or names[ci]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                if cls is None:
                    skipped_cls[names[ci]] += 1
                    continue
                if min(bw * W, bh * H) < a.min_px:
                    skipped_small += 1
                    continue
                crop = square_crop(img, cx * W, cy * H, bw * W, bh * H, a.margin)
                d = out / cls
                d.mkdir(exist_ok=True)
                imwrite(d / f"{stem}_{j:03d}.png", crop)
                counts[cls] += 1
            (frames / "labels" / f"{stem}.txt").write_text("\n".join(rows), encoding="utf-8")

        print(f"Recortes gravados em {out}: {dict(counts)}")
        if skipped_small:
            print(f"  {skipped_small} caixas menores que {a.min_px}px ignoradas")
        if skipped_cls:
            print(f"  caixas de classes ignoradas: {dict(skipped_cls)}")
        print(f"Quadros inteiros: {frames}")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
