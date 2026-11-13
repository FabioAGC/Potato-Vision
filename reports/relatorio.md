# Relatório de treino – detecção de corpos estranhos

Gerado em 2026-10-23 09:57

## Dados

- Imagens encontradas: **4145**
- Duplicatas removidas (cópias exatas + giradas/espelhadas): **3388**
- Imagens únicas usadas: **757**

| Classe | Únicas | Treino | Validação | Teste |
|---|---|---|---|---|
| Batata danificada | 110 | 78 | 16 | 16 |
| Batata | 560 | 392 | 84 | 84 |
| Planta/Galho | 44 | 30 | 7 | 7 |
| Pedra | 43 | 31 | 6 | 6 |

*Treino balanceado com rotações/espelhamentos das classes menores, mais janelas desalinhadas geradas de mosaicos: train: 3000, val: 600, test: 1500.*

## Resultado principal – validação cruzada (5 partes) – recortes centralizados

Cada imagem é testada exatamente uma vez por um modelo que não a viu no treino. É a medida mais confiável com um dataset deste tamanho. Entre parênteses: intervalo de confiança de 95%.

Avaliado em **757** imagens (87 corpos estranhos, 670 batatas).

| Modelo | Corpos estranhos detectados (recall) | Falso alarme em batata | Precisão |
|---|---|---|---|
| YOLO-cls sozinho | 96.6% (90–99%) | 0.7% (0–2%) | 94.4% |
| PaDiM sozinho | 63.2% (53–73%) | 1.3% (1–3%) | 85.9% |
| **Híbrido (usado)** | 95.4% (89–98%) | 0.3% (0–1%) | 97.6% |

- Acurácia do classificador (4 classes): **89.3%**
- AUROC do PaDiM (batata × corpo estranho): **0.978**

| Classe | n | Classificador acerta | Híbrido alarma | …como Desconhecido |
|---|---|---|---|---|
| Batata danificada | 110 | 61.8% | 1.8% | 0.9% |
| Batata | 560 | 93.6% | 0.0% | 0.0% |
| Planta/Galho | 44 | 100.0% | 100.0% | 0.0% |
| Pedra | 43 | 93.0% | 90.7% | 7.0% |

Matriz de confusão do classificador (linha = real, coluna = previsto):

| | Batata danificada | Batata | Planta/Galho | Pedra |
|---|---|---|---|---|
| **Batata danificada** | 68 | 38 | 1 | 3 |
| **Batata** | 35 | 524 | 0 | 1 |
| **Planta/Galho** | 0 | 0 | 44 | 0 |
| **Pedra** | 3 | 0 | 0 | 40 |

## Validação cruzada – janelas desalinhadas

Janelas cortadas em posição aleatória sobre mosaicos de imagens do teste, como a janela deslizante vê o quadro da câmera (o objeto quase nunca está centralizado).

Avaliado em **3000** imagens (1173 corpos estranhos, 1827 batatas).

| Modelo | Corpos estranhos detectados (recall) | Falso alarme em batata | Precisão |
|---|---|---|---|
| YOLO-cls sozinho | 94.2% (93–95%) | 2.6% (2–3%) | 95.9% |
| PaDiM sozinho | 57.6% (55–60%) | 6.4% (5–8%) | 85.2% |
| **Híbrido (usado)** | 91.4% (90–93%) | 2.0% (1–3%) | 96.8% |

- Acurácia do classificador (4 classes): **80.4%**
- AUROC do PaDiM (batata × corpo estranho): **0.880**

| Classe | n | Classificador acerta | Híbrido alarma | …como Desconhecido |
|---|---|---|---|---|
| Batata danificada | 890 | 62.5% | 3.0% | 1.1% |
| Batata | 937 | 82.9% | 1.0% | 0.3% |
| Planta/Galho | 580 | 98.4% | 98.1% | 2.6% |
| Pedra | 593 | 85.5% | 84.8% | 6.2% |

Matriz de confusão do classificador (linha = real, coluna = previsto):

| | Batata danificada | Batata | Planta/Galho | Pedra |
|---|---|---|---|---|
| **Batata danificada** | 556 | 300 | 7 | 27 |
| **Batata** | 147 | 777 | 4 | 9 |
| **Planta/Galho** | 3 | 2 | 571 | 4 |
| **Pedra** | 43 | 20 | 23 | 507 |

## Modelo final – conjunto de teste – recortes centralizados

Modelo salvo em `models/`. Conjunto de teste pequeno: os números variam bastante de uma divisão para outra.

Avaliado em **113** imagens (13 corpos estranhos, 100 batatas).

| Modelo | Corpos estranhos detectados (recall) | Falso alarme em batata | Precisão |
|---|---|---|---|
| YOLO-cls sozinho | 100.0% (77–100%) | 2.0% (1–7%) | 86.7% |
| PaDiM sozinho | 53.8% (29–77%) | 0.0% (0–4%) | 100.0% |
| **Híbrido (usado)** | 100.0% (77–100%) | 1.0% (0–5%) | 92.9% |

- Acurácia do classificador (4 classes): **89.4%**
- AUROC do PaDiM (batata × corpo estranho): **0.981**

| Classe | n | Classificador acerta | Híbrido alarma | …como Desconhecido |
|---|---|---|---|---|
| Batata danificada | 16 | 68.8% | 6.2% | 0.0% |
| Batata | 84 | 91.7% | 0.0% | 0.0% |
| Planta/Galho | 7 | 100.0% | 100.0% | 0.0% |
| Pedra | 6 | 100.0% | 100.0% | 0.0% |

Matriz de confusão do classificador (linha = real, coluna = previsto):

| | Batata danificada | Batata | Planta/Galho | Pedra |
|---|---|---|---|---|
| **Batata danificada** | 11 | 4 | 0 | 1 |
| **Batata** | 6 | 77 | 1 | 0 |
| **Planta/Galho** | 0 | 0 | 7 | 0 |
| **Pedra** | 0 | 0 | 0 | 6 |

## Modelo final – conjunto de teste – janelas desalinhadas

Janelas cortadas em posição aleatória sobre mosaicos de imagens do teste, como a janela deslizante vê o quadro da câmera (o objeto quase nunca está centralizado).

Avaliado em **1500** imagens (554 corpos estranhos, 946 batatas).

| Modelo | Corpos estranhos detectados (recall) | Falso alarme em batata | Precisão |
|---|---|---|---|
| YOLO-cls sozinho | 99.8% (99–100%) | 6.7% (5–8%) | 89.8% |
| PaDiM sozinho | 65.7% (62–70%) | 6.6% (5–8%) | 85.4% |
| **Híbrido (usado)** | 98.0% (96–99%) | 5.0% (4–7%) | 92.0% |

- Acurácia do classificador (4 classes): **80.1%**
- AUROC do PaDiM (batata × corpo estranho): **0.902**

| Classe | n | Classificador acerta | Híbrido alarma | …como Desconhecido |
|---|---|---|---|---|
| Batata danificada | 478 | 54.8% | 7.3% | 0.6% |
| Batata | 468 | 85.5% | 2.6% | 1.1% |
| Planta/Galho | 292 | 96.9% | 97.6% | 4.5% |
| Pedra | 262 | 97.7% | 98.5% | 5.7% |

Matriz de confusão do classificador (linha = real, coluna = previsto):

| | Batata danificada | Batata | Planta/Galho | Pedra |
|---|---|---|---|---|
| **Batata danificada** | 262 | 165 | 0 | 51 |
| **Batata** | 56 | 400 | 1 | 11 |
| **Planta/Galho** | 0 | 0 | 283 | 9 |
| **Pedra** | 0 | 1 | 5 | 256 |

## Desempenho

- Treino do classificador: 0 s · PaDiM: 10 s (CPU)
- Inferência: **6.3 ms por recorte** (os dois modelos, CPU)
- Limiar PaDiM calibrado para 5% de falso alarme na validação: 13.02

## Arquivos

- `models/classifier.pt`, `models/padim.pt`
- `models\classifier.onnx`
- `models\padim.onnx`