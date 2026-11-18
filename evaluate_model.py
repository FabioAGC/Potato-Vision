"""Mede a acurácia de modelos JÁ TREINADOS numa pasta de imagens rotuladas,
sem retreinar. Serve para comparar o modelo atual com o novo nas MESMAS imagens.

    python evaluate_model.py --source data/captures_rotuladas
    python evaluate_model.py --config config.yaml config.kaggle.yaml --source pasta_teste
    python evaluate_model.py --config config.kaggle.yaml --source pasta --errors reports/erros

A pasta segue o formato do dataset: uma subpasta por classe (Good, Damaged, Stone,
Plant...). Classes que o modelo não conhece (ex.: Plastico) contam como corpo
estranho: mede se o sistema alarma objetos nunca vistos.

IMPORTANTE: use imagens que o modelo NÃO viu no treino (de preferência, da sua
câmera na linha). O script avisa se encontrar imagens iguais às do dataset de treino.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from potato_cv.classifier import Classifier
from potato_cv.config import load_config, resolve
from potato_cv.data import _canonical_hash, imread, scan_source
from potato_cv.decision import OK, decide
from potato_cv.evaluate import compute_metrics, fmt_ci, fmt_pct
from potato_cv.padim import PaDiM


def load_models(cfg, engine: str):
    cls_w = resolve(cfg.classifier.weights)
    if engine == "onnx" and cls_w.with_suffix(".onnx").exists():
        cls_w = cls_w.with_suffix(".onnx")
    return Classifier(cls_w, cfg.classifier.imgsz), PaDiM.load(resolve(cfg.padim.weights), engine=engine)


def training_hashes(cfg) -> set[str]:
    """Hashes das imagens do dataset de treino do config (para avisar sobre vazamento)."""
    src = cfg.data.source
    out = set()
    for s in (src if isinstance(src, list) else [src]):
        p = resolve(s)
        if p.exists():
            for f in p.rglob("*"):
                if f.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"} and not any(x.startswith("_") for x in f.parts):
                    img = imread(f)
                    if img is not None:
                        out.add(_canonical_hash(img))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", default=["config.yaml"], help="um ou mais configs (um por modelo)")
    ap.add_argument("--source", nargs="+", required=True, help="pasta(s) com subpastas por classe")
    ap.add_argument("--engine", default="torch", choices=["torch", "onnx"])
    ap.add_argument("--errors", default=None, help="pasta onde salvar as imagens erradas (para inspecionar)")
    ap.add_argument("--no-leak-check", action="store_true", help="não compara com o dataset de treino")
    a = ap.parse_args()

    per_class, info = scan_source([Path(s) for s in a.source])
    items = [(p, c) for c, ps in per_class.items() for p in ps]
    imgs = [imread(p) for p, _ in items]
    y = [c for _, c in items]
    print(f"Imagens: {info['files_found']} encontradas, {len(items)} únicas -> {info['unique_per_class']}")

    rows = []
    for cfg_path in a.config:
        cfg = load_config(cfg_path)
        normal = set(cfg.normal_classes)
        if not a.no_leak_check:
            th = training_hashes(cfg)
            leaked = sum(_canonical_hash(im) in th for im in imgs)
            if leaked:
                print(f"\n[{cfg_path}] ATENÇÃO: {leaked} de {len(imgs)} imagens estão no dataset de treino "
                      f"({cfg.data.source}). O resultado fica otimista; use imagens novas.")
        clf, padim = load_models(cfg, a.engine)
        t0 = time.time()
        probs = clf.predict(imgs)
        scores = padim.score(imgs)
        ms = (time.time() - t0) / max(1, len(imgs)) * 1000
        d = cfg.decision
        decs = decide(probs, scores, clf.names, normal, padim.threshold, d.foreign_conf, d.normal_conf)
        m = compute_metrics(y, probs, scores, decs, clf.names, normal, padim.threshold)
        rows.append((cfg_path, m, ms))

        names = cfg.display_names
        h = m["hibrido"]
        print(f"\n=== {cfg_path}  ({cfg.classifier.weights}, {cfg.padim.weights}) ===")
        print(f"  Corpos estranhos detectados (recall): {fmt_pct(h['recall_corpo_estranho'])}{fmt_ci(h['recall_ic95'])}"
              f"  [{h['detectados']} de {m['n_corpos_estranhos']}]")
        print(f"  Falso alarme em batata:               {fmt_pct(h['falso_alarme_batata'])}{fmt_ci(h['falso_alarme_ic95'])}"
              f"  [{h['falsos_alarmes']} de {m['n_batatas']}]")
        print(f"  Precisão dos alarmes:                 {fmt_pct(h['precisao'])}")
        print(f"  Acurácia do classificador (classes):  {fmt_pct(m['classificador']['acuracia_4_classes'])}")
        if m["padim"]["auroc"] is not None:
            print(f"  AUROC do PaDiM:                       {m['padim']['auroc']:.3f}")
        print(f"  Tempo: {ms:.1f} ms por recorte (os dois modelos, {a.engine}, esta máquina)")
        print("  Por classe:  n | classificador acerta | sistema alarma | ...como Desconhecido")
        for c, v in m["por_classe"].items():
            print(f"    {names.get(c, c):<18} {v['n']:>4} | {fmt_pct(v['classificador_acerto']):>8} | "
                  f"{fmt_pct(v['hibrido_alarme']):>8} | {fmt_pct(v['hibrido_unknown']):>8}")
        cm = m["classificador"]["matriz_confusao"]
        print("  Matriz de confusão (linha = real, coluna = previsto):")
        short = [names.get(c, c)[:12] for c in cm["labels"]]
        print("    " + " " * 13 + "".join(f"{s:>13}" for s in short))
        for s, r in zip(short, cm["matriz"]):
            print(f"    {s:>13}" + "".join(f"{x:>13}" for x in r))

        if a.errors:
            out = Path(a.errors) / Path(cfg_path).stem
            if out.exists():
                shutil.rmtree(out)
            n_err = 0
            for (p, c), im, dec, sc in zip(items, imgs, decs, scores):
                foreign = c not in normal
                alarm = dec.status != OK
                if foreign and not alarm:
                    kind = "perdido"           # corpo estranho que passou
                elif not foreign and alarm:
                    kind = "falso_alarme"      # batata que gerou alarme
                else:
                    continue
                dst = out / kind / c
                dst.mkdir(parents=True, exist_ok=True)
                tag = f"{dec.label}_{dec.confidence:.2f}_anom{sc / padim.threshold:.2f}"
                cv2.imwrite(str(dst / f"{p.stem}__{tag}.png"), im)
                n_err += 1
            print(f"  {n_err} erros salvos em {out}/ (perdido/ e falso_alarme/)")

    if len(rows) > 1:
        print("\n=== Comparação (mesmas imagens) ===")
        print(f"  {'config':<24} {'recall':>8} {'falso al.':>10} {'precisão':>9} {'acurácia':>9} {'ms/rec':>7}")
        for cfg_path, m, ms in rows:
            h = m["hibrido"]
            print(f"  {cfg_path:<24} {fmt_pct(h['recall_corpo_estranho']):>8} {fmt_pct(h['falso_alarme_batata']):>10} "
                  f"{fmt_pct(h['precisao']):>9} {fmt_pct(m['classificador']['acuracia_4_classes']):>9} {ms:>7.1f}")
        print("  Diferenças pequenas com poucas imagens podem ser acaso: olhe os intervalos (95%) acima.")


if __name__ == "__main__":
    main()
