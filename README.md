# Potato CV – Detecção de corpos estranhos na linha de batatas

Detecta **pedras**, **plantas/galhos** e **dejetos desconhecidos** (plástico, madeira, metal…)
em imagens da esteira. Batata boa e batata danificada são tratadas como normais.

## Como funciona

```
 câmera ──► quadro ──► janela deslizante (1 objeto por janela)
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
     YOLO-cls (Ultralytics)        PaDiM (anomalia)
  "é Pedra / Planta / Batata?"   "isto parece batata normal?"
             └─────────────┬─────────────┘
                           ▼
                    regra de decisão
     Pedra/Planta com confiança  ───────────────► VERMELHO (corpo estranho)
     PaDiM anormal + classificador sem certeza ─► AMARELO  (desconhecido)
     caso contrário ────────────────────────────► OK
```

* **PaDiM** aprende só com batatas normais: qualquer coisa diferente gera nota de
  anomalia alta, por isso pega dejetos que nunca apareceram no treino.
* **YOLO-cls** dá o nome do objeto quando ele é uma classe conhecida.

Três cuidados para funcionar no quadro inteiro (e não só em recortes centralizados):

1. **Janelas desalinhadas no treino** – a janela deslizante quase nunca pega o objeto
   centralizado. O treino gera milhares de janelas cortadas de mosaicos de imagens em
   posição aleatória, para os modelos aprenderem esse caso.
2. **Média por região (`inference.aggregation: grid`)** – cada região do quadro recebe a
   média de todas as janelas que a cobrem; erro isolado de uma janela se dilui.
3. **Confirmação temporal (`inference.confirm_frames`)** – o alarme só aparece se
   persistir em N quadros seguidos; cada objeto é contado uma vez.

## Ajuste fino (config.yaml)

| Parâmetro | Efeito |
|---|---|
| `decision.foreign_conf` | ↓ pega mais corpos estranhos, ↑ menos falso alarme (padrão 0.8, escolhido na validação) |
| `decision.normal_conf` | ↑ mais "Desconhecidos" |
| `padim.target_fpr` | fração de batatas normais acima do limiar do PaDiM na calibração |
| `inference.confirm_frames` | ↑ menos alarmes piscando; deve ser menor que o nº de quadros em que um objeto fica visível |
| `inference.tile_px` | tamanho (px) de **uma batata** no quadro da câmera |
| `inference.roi` | `[x, y, largura, altura]` só da esteira – áreas fora dela geram "Desconhecido" |

## Instalação

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
$py = "$env:USERPROFILE\.venvs\potato-cv\Scripts\python.exe"
```

O ambiente fica em `%USERPROFILE%\.venvs\potato-cv` (fora do OneDrive, para não sincronizar ~1 GB).

## Uso

| Tarefa | Comando |
|---|---|
| Treinar (dataset do config) | `& $py train.py` |
| Treinar com outro dataset | `& $py train.py --source D:\fotos_batata` |
| Treinar + métricas confiáveis | `& $py train.py --cv 5 --unknown-test` |
| Treinar na GPU (config maior) | `& $py train.py --config config.kaggle.yaml` |
| Tela ao vivo (simulador) | `& $py run.py` |
| Tela ao vivo (webcam USB) | `& $py run.py --camera usb --index 0` |
| Tela ao vivo (câmera IP) | `& $py run.py --camera rtsp --url rtsp://user:senha@IP:554/stream` |
| Teste sem janela | `& $py run.py --headless --frames 300` |
| Capturar imagens da câmera | `& $py capture.py --every 1` |

Teclas na tela ao vivo: **q** sair · **p** pausar · **s** salvar quadro.

### Treinar em GPU (Kaggle)

`kaggle_train.ipynb` treina com a configuração maior de `config.kaggle.yaml`
(YOLO `s` em 128 px com RandAugment, PaDiM com `wide_resnet50_2`, mais janelas sintéticas):

1. No Kaggle: *Create → New Notebook → File → Import Notebook* e escolha `kaggle_train.ipynb`.
2. *Add Input → Upload*: envie `dataset_potatoes.zip` como Dataset (opcional, o repositório já traz uma cópia).
3. *Settings*: Accelerator = GPU (T4 ou P100), Internet = ligado. *Run All*.
4. Baixe `potato_models_gpu.zip` da aba Output, descompacte na pasta do projeto e rode
   `& $py run.py --config config.kaggle.yaml`.

Os modelos de GPU ficam em `models/gpu/` e não substituem os de CPU. Compare as duas
validações cruzadas no fim do notebook antes de trocar. Para treinar em GPU fora do Kaggle:
`train.py --config config.kaggle.yaml` (`device: auto` usa a GPU se houver).

### Juntar um dataset do Roboflow

Datasets de **detecção** (quadro inteiro com caixas) do Roboflow viram recortes por classe com
`import_roboflow.py`. Exporte em **YOLOv8** (*Download Dataset → YOLOv8 → download zip*) e rode:

```powershell
& $py import_roboflow.py potato.v1i.yolov8.zip                 # cria dataset_roboflow/<Classe>/
& $py train.py --source dataset_potatoes/dataset_potatoes dataset_roboflow
```

* As classes são mapeadas pelo nome (potato → Good, stone/rock → Stone, plant/leaf → Plant,
  damaged/rotten → Damaged). Outras são ignoradas; use `--map nome=Classe` para ajustar.
* Cópias aumentadas pelo Roboflow (`*.rf.*`) viram uma só foto, para não vazar do treino para o teste.
* Os quadros inteiros ficam em `dataset_roboflow/_frames/` (ignorados no treino).
* No Kaggle o notebook baixa sozinho [acu/potato-sjlt6](https://universe.roboflow.com/acu/potato-sjlt6)
  se houver o secret `ROBOFLOW_API_KEY`.

### Formato do dataset

Uma pasta por classe (pode haver pastas de split no meio, elas são ignoradas):

```
meu_dataset/
  Good/     *.bmp|png|jpg      <- normal
  Damaged/  ...                <- normal  (lista em normal_classes no config)
  Stone/    ...                <- corpo estranho
  Plant/    ...                <- corpo estranho
  Plastico/ ...                <- nova classe? basta criar a pasta
```

O treino remove automaticamente duplicatas (inclusive cópias giradas/espelhadas),
redivide treino/validação/teste e gera `reports/relatorio.md`.

## Câmera

Tudo em `config.yaml` → `camera.type`:

| Tipo | Uso | Observação |
|---|---|---|
| `simulator` | Sem hardware | Esteira montada com imagens do conjunto de teste |
| `usb` | Webcam / UVC | `camera.usb.index` |
| `rtsp` | Câmera IP | `camera.rtsp.url` |
| `genicam` | Industrial GigE / USB3 (Basler, Hikrobot, FLIR…) | `uv pip install harvesters` + `.cti` do SDK do fabricante |
| `video` / `images` | Reproduzir gravações | |

As câmeras ao vivo leem numa thread separada (sempre o quadro mais novo) e
**reconectam sozinhas** se o sinal cair.

## Checklist quando a câmera chegar

1. Configurar `camera.type` e testar com `run.py`.
2. Medir quantos pixels uma batata ocupa no quadro → `inference.tile_px`.
3. Definir `inference.roi` para olhar só a esteira.
4. Rodar a linha **só com batata boa** e capturar recortes:
   `capture.py --crops --label Good --every 0.5` (alguns milhares de recortes).
5. Capturar também alguns exemplos de pedra/galho/dejetos e colocar nas pastas
   correspondentes.
6. Retreinar com `train.py --source data/captures` (pode juntar com o dataset atual).
7. Ajustar `padim.target_fpr` e `decision.*` conforme a tolerância a falso alarme.

## Estrutura

```
config.yaml          configuração geral
config.kaggle.yaml   configuração maior para treino em GPU
kaggle_train.ipynb   notebook de treino no Kaggle
import_roboflow.py   converte export YOLOv8 do Roboflow em recortes por classe
train.py             pipeline de treino + avaliação + exportação
run.py               detecção ao vivo com tela
capture.py           captura de imagens para dataset
potato_cv/
  data.py            limpeza, deduplicação, split
  classifier.py      YOLO-cls (Ultralytics)
  padim.py           PaDiM (detecção de anomalia)
  decision.py        regra de fusão
  detector.py        janela deslizante no quadro + desenho
  camera.py          camadas de câmera (USB, RTSP, GenICam, vídeo, imagens)
  simulator.py       esteira simulada
  evaluate.py        métricas
models/              modelos treinados (.pt e .onnx)
reports/             relatório de treino
```
