"""Pipeline de treino completo.

    python train.py                       # usa o dataset do config.yaml
    python train.py --source D:/fotos     # outro dataset (pastas por classe)
    python train.py --source dataset_potatoes/dataset_potatoes dataset_roboflow   # vários datasets juntos
    python train.py --cv 5                # + validação cruzada (métrica mais confiável)
    python train.py --unknown-test        # + teste de objeto desconhecido
    python train.py --device cuda         # treina na GPU (padrão: config `device`, "auto")
    python train.py --config config.kaggle.yaml   # configuração maior para GPU (Kaggle)

Etapas: prepara dados -> treina YOLO-cls -> treina PaDiM -> calibra limiar ->
avalia no teste -> exporta ONNX -> gera reports/relatorio.md
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np

from potato_cv.classifier import Classifier, train_classifier
from potato_cv.config import ROOT, load_config, resolve, resolve_device
from potato_cv.data import (kfold, list_split, load_images, merge_groups, prepare_dataset, scan_source,
                            write_parts)
from potato_cv.decision import decide
from potato_cv.evaluate import compute_metrics, fmt_ci, fmt_pct
from potato_cv.padim import PaDiM

log = logging.getLogger("train")


# ---------------------------------------------------------------- um experimento
def run_experiment(cfg, data_dir: Path, out_dir: Path, tag: str, exclude: set[str] = frozenset(),
                   epochs: int | None = None, reuse: bool = False) -> dict:
    """Treina classificador + PaDiM em data_dir/{train,val} e avalia em data_dir/test.
    `exclude`: classes retiradas do treino do classificador (teste de desconhecido)."""
    normal = set(cfg.normal_classes)
    out_dir.mkdir(parents=True, exist_ok=True)

    cls_data = data_dir
    if exclude:  # cópia sem as classes excluídas
        cls_data = out_dir / "cls_data"
        if cls_data.exists():
            shutil.rmtree(cls_data)
        for s in ("train", "val"):
            for d in (data_dir / s).iterdir():
                if d.is_dir() and d.name not in exclude:
                    shutil.copytree(d, cls_data / s / d.name)

    c = cfg.classifier
    device = cfg.device
    t0 = time.time()
    w = out_dir / "classifier.pt"
    if not (reuse and w.exists()):
        train_classifier(cls_data, w, c.model, c.imgsz, epochs or c.epochs, c.batch, c.patience, name=tag,
                         device=device, workers=int(c.get("workers", 0)), auto_augment=c.get("auto_augment"),
                         extra=dict(c.get("extra") or {}))
    else:
        log.info("    reaproveitando classificador já treinado: %s", w)
    t_cls = time.time() - t0
    clf = Classifier(w, c.imgsz, device=device)

    p = cfg.padim
    t0 = time.time()
    padim = PaDiM(p.backbone, p.input_size, p.feature_dims, p.get("score_mode", "mean"), device=device)
    # PaDiM aprende batata normal: recortes originais + janelas desalinhadas
    normal_train = [x for x, k in list_split(data_dir, "train", kind="orig+win") if k in normal]
    padim.fit(load_images(normal_train))
    val_normal = [x for x, k in list_split(data_dir, "val", kind="orig+win") if k in normal]
    padim.calibrate(padim.score(load_images(val_normal)), p.target_fpr)
    padim.save(out_dir / "padim.pt")
    t_pad = time.time() - t0

    d = cfg.decision
    results = {}
    for kind in ("orig", "win"):
        test = list_split(data_dir, "test", kind=kind)
        if not test:
            continue
        imgs = load_images([x for x, _ in test])
        y = [k for _, k in test]
        t0 = time.time()
        probs = clf.predict(imgs)
        scores = padim.score(imgs)
        ms = (time.time() - t0) / max(1, len(imgs)) * 1000
        decs = decide(probs, scores, clf.names, normal, padim.threshold, d.foreign_conf, d.normal_conf)
        m = compute_metrics(y, probs, scores, decs, clf.names, normal, padim.threshold)
        m["_raw"] = {"y": y, "probs": probs.tolist(), "scores": scores.tolist(), "names": clf.names,
                     "threshold": padim.threshold}
        results[kind] = m
    m = results["orig"]
    m["janelas"] = results.get("win")
    m["tempo"] = {"treino_classificador_s": round(t_cls, 1), "treino_padim_s": round(t_pad, 1),
                  "inferencia_ms_por_recorte": round(ms, 2)}
    return m


# ---------------------------------------------------------------- validação cruzada
def cross_validate(cfg, k: int, source: Path | list[Path]) -> dict:
    """Cada imagem é testada uma vez, por um modelo que não a viu no treino."""
    if k < 3:
        raise SystemExit("--cv precisa de pelo menos 3 partes (teste, validação e treino).")
    per_class, _ = scan_source(source)
    folds = kfold(per_class, k, cfg.data.seed)
    normal = set(cfg.normal_classes)
    win = dict(cfg.data.get("windows") or {})
    acc = {"orig": ([], [], []), "win": ([], [], [])}
    names = None
    for i in range(k):
        log.info("Validação cruzada: parte %d/%d", i + 1, k)
        parts = {"test": folds[i], "val": folds[(i + 1) % k],
                 "train": merge_groups(*[folds[j] for j in range(k) if j not in (i, (i + 1) % k)])}
        dd = ROOT / "data" / "cv" / f"fold{i}"
        write_parts(parts, dd, normal, {s_: max(1, n // k * 2) if s_ == "test" else n for s_, n in win.items()},
                    cfg.data.seed + 100 * i)
        m = run_experiment(cfg, dd, ROOT / "runs" / "cv" / f"fold{i}", f"cv_fold{i}", epochs=cfg.classifier.get("cv_epochs"))
        for kind, mm in (("orig", m), ("win", m.get("janelas"))):
            if not mm:
                continue
            r = mm["_raw"]
            names = r["names"]
            ys, pr, sc = acc[kind]
            ys += r["y"]
            pr.append(np.array(r["probs"]))
            # nota normalizada pelo limiar do próprio fold (limiares diferem entre partes)
            sc.append(np.array(r["scores"]) / r["threshold"])
    dc = cfg.decision
    out = {}
    for kind, (ys, pr, sc) in acc.items():
        if not ys:
            continue
        pr, sc = np.concatenate(pr), np.concatenate(sc)
        decs = decide(pr, sc, names, normal, 1.0, dc.foreign_conf, dc.normal_conf)
        out[kind] = compute_metrics(ys, pr, sc, decs, names, normal, 1.0)
    res = out["orig"]
    res["janelas"] = out.get("win")
    res["folds"] = k
    return res


# ---------------------------------------------------------------- objeto desconhecido
def unknown_test(cfg, data_dir: Path) -> dict:
    """Para cada corpo estranho conhecido (Pedra, Planta): retira a classe do
    treino do classificador e verifica se o sistema ainda alarma esses objetos
    como 'Desconhecido' - simula um dejeto que o modelo nunca viu."""
    normal = set(cfg.normal_classes)
    foreign = [d.name for d in (data_dir / "train").iterdir() if d.is_dir() and d.name not in normal]
    out = {}
    for f in foreign:
        log.info("Teste de desconhecido: classificador treinado SEM '%s'", f)
        m = run_experiment(cfg, data_dir, ROOT / "runs" / "unknown" / f"sem_{f}", f"unknown_sem_{f}", exclude={f},
                           epochs=cfg.classifier.get("cv_epochs"))
        pc = m["por_classe"].get(f, {})
        out[f] = {"n": pc.get("n"), "alarmado": pc.get("hibrido_alarme"),
                  "como_desconhecido": pc.get("hibrido_unknown"),
                  "padim_sozinho": pc.get("padim_alarme"),
                  "falso_alarme_batata": m["hibrido"]["falso_alarme_batata"]}
    return out


# ---------------------------------------------------------------- relatório
def write_report(cfg, ds: dict, main: dict, cv: dict | None, unk: dict | None, onnx: list[str]) -> Path:
    names = cfg.display_names
    L = ["# Relatório de treino – detecção de corpos estranhos", "",
         f"Gerado em {time.strftime('%Y-%m-%d %H:%M')}", "",
         "## Dados", "",
         f"- Imagens encontradas: **{ds['files_found']}**"
         + ("".join(f"\n  - `{Path(k).name}`: {v}" for k, v in ds.get("files_per_source", {}).items())
            if len(ds.get("files_per_source", {})) > 1 else ""),
         f"- Duplicatas removidas (cópias exatas + giradas/espelhadas): **{ds['duplicates_removed']}**",
         f"- Imagens únicas usadas: **{ds['unique_images'] - ds['label_conflicts_removed']}**",
         "", "| Classe | Únicas | Treino | Validação | Teste |", "|---|---|---|---|---|"]
    for c, n in ds["unique_per_class"].items():
        su = ds["splits_unique"]
        L.append(f"| {names.get(c, c)} | {n} | {su['train'].get(c)} | {su['val'].get(c)} | {su['test'].get(c)} |")
    L += ["", "*Treino balanceado com rotações/espelhamentos das classes menores, mais janelas "
          "desalinhadas geradas de mosaicos: "
          + ", ".join(f"{k.replace('_janelas', '')}: {sum(v.values())}" for k, v in ds.get("splits_written", {}).items()
                      if k.endswith("_janelas")) + ".*", ""]

    def block(title, m, note=""):
        out = [f"## {title}", ""]
        if note:
            out += [note, ""]
        out += [f"Avaliado em **{m['n_imagens']}** imagens ({m['n_corpos_estranhos']} corpos estranhos, "
                f"{m['n_batatas']} batatas).", "",
                "| Modelo | Corpos estranhos detectados (recall) | Falso alarme em batata | Precisão |",
                "|---|---|---|---|"]
        for k, lbl in (("classificador", "YOLO-cls sozinho"), ("padim", "PaDiM sozinho"), ("hibrido", "**Híbrido (usado)**")):
            b = m[k]
            out.append(f"| {lbl} | {fmt_pct(b['recall_corpo_estranho'])}{fmt_ci(b['recall_ic95'])} "
                       f"| {fmt_pct(b['falso_alarme_batata'])}{fmt_ci(b['falso_alarme_ic95'])} | {fmt_pct(b['precisao'])} |")
        out += ["", f"- Acurácia do classificador (4 classes): **{fmt_pct(m['classificador']['acuracia_4_classes'])}**",
                f"- AUROC do PaDiM (batata × corpo estranho): **{m['padim']['auroc']:.3f}**" if m['padim']['auroc'] else "",
                "", "| Classe | n | Classificador acerta | Híbrido alarma | …como Desconhecido |", "|---|---|---|---|---|"]
        for c, v in m["por_classe"].items():
            out.append(f"| {names.get(c, c)} | {v['n']} | {fmt_pct(v['classificador_acerto'])} | "
                       f"{fmt_pct(v['hibrido_alarme'])} | {fmt_pct(v['hibrido_unknown'])} |")
        cm = m["classificador"]["matriz_confusao"]
        out += ["", "Matriz de confusão do classificador (linha = real, coluna = previsto):", "",
                "| | " + " | ".join(names.get(c, c) for c in cm["labels"]) + " |",
                "|---" * (len(cm["labels"]) + 1) + "|"]
        for c, row in zip(cm["labels"], cm["matriz"]):
            out.append(f"| **{names.get(c, c)}** | " + " | ".join(map(str, row)) + " |")
        return out + [""]

    win_note = ("Janelas cortadas em posição aleatória sobre mosaicos de imagens do teste, como a janela "
                "deslizante vê o quadro da câmera (o objeto quase nunca está centralizado).")
    if cv:
        L += block(f"Resultado principal – validação cruzada ({cv['folds']} partes) – recortes centralizados", cv,
                   "Cada imagem é testada exatamente uma vez por um modelo que não a viu no treino. "
                   "É a medida mais confiável com um dataset deste tamanho. Entre parênteses: intervalo de confiança de 95%.")
        if cv.get("janelas"):
            L += block("Validação cruzada – janelas desalinhadas", cv["janelas"], win_note)
    L += block("Modelo final – conjunto de teste – recortes centralizados", main,
               f"Modelo salvo em `{Path(cfg.classifier.weights).parent.as_posix()}/`. Conjunto de teste pequeno: os números variam bastante de uma divisão para outra.")
    if main.get("janelas"):
        L += block("Modelo final – conjunto de teste – janelas desalinhadas", main["janelas"], win_note)
    if unk:
        L += ["## Teste de objeto desconhecido", "",
              "Simula um dejeto nunca visto: o classificador é treinado **sem** uma das classes e "
              "verificamos se o sistema ainda alarma esses objetos.", "",
              "| Classe retirada do treino | n teste | Alarmado pelo híbrido | …como Desconhecido | PaDiM sozinho | Falso alarme batata |",
              "|---|---|---|---|---|---|"]
        for c, v in unk.items():
            L.append(f"| {names.get(c, c)} | {v['n']} | {fmt_pct(v['alarmado'])} | {fmt_pct(v['como_desconhecido'])} | "
                     f"{fmt_pct(v['padim_sozinho'])} | {fmt_pct(v['falso_alarme_batata'])} |")
        L.append("")
    t = main["tempo"]
    dev = "GPU" if str(cfg.device).startswith("cuda") else "CPU"
    L += ["## Desempenho", "",
          f"- Config: `{cfg.get('_path', 'config.yaml')}` · classificador `{cfg.classifier.model}` "
          f"imgsz {cfg.classifier.imgsz} · PaDiM `{cfg.padim.backbone}` {cfg.padim.input_size}px",
          f"- Treino do classificador: {t['treino_classificador_s']:.0f} s · PaDiM: {t['treino_padim_s']:.0f} s ({dev})",
          f"- Inferência: **{t['inferencia_ms_por_recorte']:.1f} ms por recorte** (os dois modelos, {dev}"
          + ("; na máquina da linha, em CPU, será mais lento)" if dev == "GPU" else ")"),
          f"- Limiar PaDiM calibrado para {100 * cfg.padim.target_fpr:.0f}% de falso alarme na validação: {main['padim']['limiar']:.2f}",
          "", "## Arquivos", "", f"- `{cfg.classifier.weights}`, `{cfg.padim.weights}`"] + [f"- `{o}`" for o in onnx]
    p = ROOT / "reports" / "relatorio.md"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(L), encoding="utf-8")
    return p


def _strip(m: dict | None) -> dict | None:
    """Remove os dados brutos antes de salvar o JSON."""
    if m is None:
        return None
    return {k: (_strip(v) if k == "janelas" else v) for k, v in m.items() if k != "_raw"}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Treina o detector de corpos estranhos.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", nargs="+", default=None,
                    help="pasta(s) do dataset (sobrescreve o config); várias pastas são juntadas")
    ap.add_argument("--cv", type=int, default=0, help="nº de partes da validação cruzada (0 = não)")
    ap.add_argument("--unknown-test", action="store_true", help="teste de objeto desconhecido")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="reaproveita o classificador de runs/final (não retreina)")
    ap.add_argument("--skip-prepare", action="store_true", help="usa data/prepared como está")
    ap.add_argument("--device", default=None, help="auto | cpu | cuda (sobrescreve o config)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    cfg = load_config(a.config)
    cfg["device"] = resolve_device(a.device or cfg.get("device", "auto"))
    cfg["_path"] = Path(a.config).name if a.config else "config.yaml"
    log.info("Dispositivo de treino: %s", cfg["device"])
    src = a.source or cfg.data.source
    source = [resolve(x) for x in src] if isinstance(src, (list, tuple)) else resolve(src)
    if isinstance(source, list) and len(source) == 1:
        source = source[0]
    prepared = resolve(cfg.data.prepared)

    log.info("1/5 Preparando dados de %s", ", ".join(map(str, source)) if isinstance(source, list) else source)
    rep_json = prepared / "dataset_report.json"
    if a.skip_prepare or a.reuse:
        if not rep_json.exists():
            raise SystemExit("data/prepared não existe - rode sem --skip-prepare/--reuse")
        ds = json.loads(rep_json.read_text(encoding="utf-8"))
    else:
        ds = prepare_dataset(source, prepared, tuple(cfg.data.split), cfg.data.seed,
                             set(cfg.normal_classes), dict(cfg.data.get("windows") or {}))
    log.info("    %d imagens -> %d únicas por classe: %s", ds["files_found"], ds["unique_images"], ds["unique_per_class"])

    log.info("2/5 Treinando modelo final (YOLO-cls + PaDiM)")
    main_m = run_experiment(cfg, prepared, ROOT / "runs" / "final", "final", reuse=a.reuse)
    resolve(cfg.classifier.weights).parent.mkdir(parents=True, exist_ok=True)
    resolve(cfg.padim.weights).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "runs" / "final" / "classifier.pt", resolve(cfg.classifier.weights))
    shutil.copy(ROOT / "runs" / "final" / "padim.pt", resolve(cfg.padim.weights))
    log.info("    híbrido no teste: recall %s, falso alarme %s",
             fmt_pct(main_m["hibrido"]["recall_corpo_estranho"]), fmt_pct(main_m["hibrido"]["falso_alarme_batata"]))

    onnx = []
    if not a.no_export:
        log.info("3/5 Exportando ONNX")
        try:
            onnx.append(str(Classifier(resolve(cfg.classifier.weights), cfg.classifier.imgsz).export_onnx().relative_to(ROOT)))
            onnx.append(str(PaDiM.load(resolve(cfg.padim.weights)).export_onnx(resolve(cfg.padim.weights)).relative_to(ROOT)))
        except Exception as e:
            log.warning("    Falha na exportação ONNX: %s", e)

    cv = None
    if a.cv:
        log.info("4/5 Validação cruzada com %d partes", a.cv)
        cv = cross_validate(cfg, a.cv, source)
    unk = None
    if a.unknown_test:
        log.info("5/5 Teste de objeto desconhecido")
        unk = unknown_test(cfg, prepared)

    metrics = {"dataset": ds, "teste": _strip(main_m), "validacao_cruzada": _strip(cv), "desconhecido": unk}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    rep = write_report(cfg, ds, main_m, cv, unk, onnx)
    log.info("Pronto. Relatório: %s", rep)


if __name__ == "__main__":
    main()
