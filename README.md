

Ajuste de hiperparámetros **con métricas no supervisadas** para detectores de anomalías
(Isolation Forest, LOF y Autoencoder) sobre **TON_IoT (Network)**, pensado para dispositivos de
borde (**Raspberry Pi 5**).

El preprocesamiento fue facilitado por el Profesor Guía Dr Juan Ignacio Iturbe-Araya
el **preprocessor  tal como está**: no se modifica
ninguno de sus `.py`. Todo lo demás es código de este proyecto.

---

## 1. Qué hay en la carpeta

```
tesis_hpo_edge/
├── preprocessor/            <- aquí va el preprocessor (ver preprocessor/LEEME.txt)
├── Datasets/
│   ├── Sources/TON-IoT/Train_Test_datasets/Train_Test_Network_dataset/
│   │                        <- aquí va Train_Test_Network.csv (ver el LEEME.txt de esa carpeta)
│   └── Parquet/             <- aquí escribe el preprocessor (train/test_*.parquet)
├── config/experiment.yaml   <- toda la configuración, con comentarios
├── src/                     <- el código del experimento
├── scripts/                 <- configurar_preprocessor.py y check_env.py
├── tests/                   <- prueba rápida con datos sintéticos
├── data/splits/             <- bloques train / val / test (los crea prepare_splits.py)
├── outputs/                 <- modelos, estudios de Optuna, resultados y tablas LaTeX
├── run_all.py               <- corre las etapas en orden
├── requirements.txt         <- librerías para el PC
└── requirements-pi.txt      <- librerías para la Raspberry
```

La carpeta tiene que estar en una ruta **sin tildes ni espacios** (ejemplo:
`C:\Users\belen\Desktop\tesis_hpo_edge`). El código del profesor lee los archivos con la codificación de Windows y las tildes le pueden dar problemas.

## 2. Por dónde pasan los datos (y quién ve las etiquetas)

| Etapa                       | Script        | ¿Usa etiquetas? |
|-----------------------------|---------------|------------------|
| Preprocesar (profesor)      | pasos 00–03   | su `train` mezcla tráfico normal y ataques |
| Bloques                     | `prepare_splits.py` | solo para separar lo normal de los ataques; **train y val tienen solo tráfico normal** |
| Selección de variables y escalado | `prepare_splits.py` | **no**: usa solo el train normal |
| Optimización no supervisada | `optimize.py` | **no** (ni siquiera abre los `test_*`) |
| Oráculo (cota superior)     | `optimize.py` | sí, pero solo de `test_oracle` |
| Medición en el borde        | `edge_benchmark.py` | no |
| Elección del modelo         | `evaluate.py` | **no**: toma el primer modelo del top-k que cumple los límites del borde; el MCC no se usa para elegir |
| Evaluación final            | `evaluate.py` | sí, de `test_final`, una sola vez (antes revisa su hash SHA-256) |

Los 3 bloques del diagrama aprobado son **train** (normal), **val** (normal) y **test**
(normal + ataques). El test se divide en `test_oracle` (lo usa solo el oráculo) y `test_final`
(lo usa solo `evaluate.py`), manteniendo la proporción de cada tipo de ataque.

- `train` y `val` salen del **tráfico normal del train del preprocessor** (sus ataques se descartan).
- `test` es **el test completo del preprocessor**. Así, los ataques de prueba nunca se usaron
  para ajustar su escalador.
- `val` se parte en dos mitades: una sirve para **fijar el umbral** (percentil `q`=0.95 de los
  puntajes normales) y en la otra se calculan las funciones objetivo.

## 3. Instalación (Windows, PowerShell)

El proyecto usa un entorno `.venv` con Python 3.10. Si ya existe, basta con activarlo; si no,
se crea así:

```powershell
cd C:\Users\belen\Desktop\tesis_hpo_edge
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\check_env.py
```

`tensorflow` es opcional: si está instalado, el autoencoder se entrena con Keras; si no, con
scikit-learn. `lightgbm` y `statsmodels` solo hacen falta si usas `features.mode: selector`
(paso 04 del profesor).

## 4. Paso a paso

**A. Preparar (una sola vez)**

1. Descomprime `preprocessor.zip` y copia el **contenido** de su carpeta `preprocessor` dentro de
   `tesis_hpo_edge\preprocessor\`. Tiene que quedar `preprocessor\preprocessor_v2\`,
   `preprocessor\config.yaml`, etc.
2. Copia `Train_Test_Network.csv` (TON_IoT, sin modificar) en
   `Datasets\Sources\TON-IoT\Train_Test_datasets\Train_Test_Network_dataset\`.
3. Corre `python scripts\configurar_preprocessor.py`. Este script deja el `config.yaml` y el
   `config.json` del profesor apuntando a esta carpeta y guarda copias `.bak_proyecto` (el
   porqué está en la sección 8).

**B. Prueba corta de todo el flujo (25 trials, 1 semilla; con Keras demora unos 15–30 min)**

```powershell
python run_all.py --rapido
```

**C. Corrida real.** Antes, revisa `config/experiment.yaml`: con `input.variant: processed` se
usa el 100 % de los datos.

```powershell
python run_all.py --etapas preprocesar,splits      # CSV -> Parquet -> bloques
python run_all.py --etapas diagnostico             # ¿sirven las métricas? (opcional, pero recomendado)
python run_all.py --etapas optimizar               # Optuna: 100 trials x 5 semillas
```

**D. En la Raspberry Pi 5**

Copia a la Pi: `src/`, `config/`, `scripts/`, `run_all.py`, `requirements-pi.txt`,
`outputs/models/` (completa) y `data/splits/edge_sample.csv`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-pi.txt          # scikit-learn y numpy: la MISMA versión que en el PC
python scripts/check_env.py --rol pi
python src/edge_benchmark.py                # mide la latencia, la CPU y la RAM de cada modelo
```

Después trae de vuelta al PC `outputs/results/edge_benchmark.csv` y `edge_platform.json`
(en la misma ruta). En `outputs/models/env.json` están las versiones con que se entrenaron los
modelos.

**E. De vuelta en el PC**

```powershell
python run_all.py --etapas evaluar,reporte
```

## 5. Qué resultados salen y dónde quedan

| Archivo | Contenido |
|---|---|
| `data/splits/manifest.json` | tamaños, tipos de ataque de cada bloque y hashes SHA-256 |
| `data/splits/features.json` | variables elegidas y cuántas se descartaron en cada filtro |
| `outputs/optuna/*.json` | un estudio por cada detector × métrica × semilla (mejor valor y top-k) |
| `outputs/results/diagnose_objectives*.csv/json` | correlación de cada métrica con el MCC y el AUROC reales |
| `outputs/results/edge_benchmark.csv` | latencia (mediana y p95), CPU al ritmo de flujos, RAM y si cumple los límites |
| `outputs/results/final_evaluation.csv` | MCC, AUROC, AUPRC, BACC, F1, Recall, FPR, FNR, MCC ajustado a otra proporción de ataques… |
| `outputs/results/balance.csv`, `resumen.md` | mejor criterio de cada detector comparado con la línea base |
| `outputs/results/recall_por_ataque.csv` | recall por tipo de ataque |
| `outputs/results/hypothesis_check.json` | si la mejor métrica cambia según el detector |
| `outputs/tables/tab_mock_*.tex` | tablas para la tesis (se incluyen con `\input{}`) |

## 6. Opciones que puedes cambiar (`config/experiment.yaml`)

- `input.variant`: `stratified` (10 % de los datos, más rápido) o `processed` (todos los datos, para la corrida final).
- `features.mode`: `unsupervised` (recomendado), `selector` (columnas del paso 04 del profesor, que usan etiquetas salvo `vif`) o `all`.
- `features.max_features` (40): con menos columnas los modelos son más livianos; con más, tienen más información.
- `threshold.q` (0.95): la tasa de falsas alarmas buscada es 1 − q. Es una regla fija, no un hiperparámetro.
- `split.test_target_prevalence`: con `null` el test queda con su proporción natural de ataques; con `0.10` se sacan ataques hasta que sean el 10 %.
- `optimization.oracle_prevalence`: con `null` el oráculo usa el MCC normal; con `0.10` optimiza el MCC calculado como si hubiera 10 % de ataques.
- `edge.*`: límites de latencia, CPU (al ritmo de `flow_rate_per_s`) y RAM.

### 6.1 Dónde se implementa cada decisión del Capítulo 3

| Sección | Decisión | Archivo y función | Parámetro (`experiment.yaml`) o constante |
|---|---|---|---|
| 3.1.3 | Preprocesador del profesor (pasos 00–03) | `run_all.py` → `etapa_preprocesar` | `input.variant` |
| 3.2.2 | Selección de variables sin etiquetas | `prepare_splits.py` → `select_unsupervised` | `features.max_features`, `min_minority_freq`, `max_abs_corr`; muestra de 20 000 filas (fija) |
| 3.2.2 | Estandarización con el train normal | `prepare_splits.py` → `fit_scaler`, `apply_scaler` | `scaling.method`, `scaling.clip` |
| 3.2.3 | Train/val (85/15 del normal) | `prepare_splits.py` → `main` | `split.val_frac_of_normal`, `split.seed` |
| 3.2.3 | Test en oráculo/final, estratificado por tipo | `prepare_splits.py` → `stratified_halves` | `split.oracle_frac_of_test` |
| 3.2.3 | Val en calibración/objetivo | `common.py` → `split_val` | `threshold.calib_frac` |
| 3.2.3 | Submuestra de 10 000 filas de train | `common.py` → `train_for`, `subsample` | `optimization.max_train_rows`, `max_train_rows_lof`; semilla 42 (fija) |
| 3.2.3 / 3.2.11 | Registro de particiones y hash SHA-256 | `prepare_splits.py` → `main` (manifest) | — |
| 3.2.4 | Detectores y puntajes | `detectors.py` → `IForestDetector`, `LOFDetector`, `AutoencoderDetector` | `autoencoder.backend` |
| 3.2.4 | Espacios de búsqueda (Tabla 3.5) | `search_space.py` → `suggest_params` | `search_space.iforest_max_samples` |
| 3.2.5 | Umbral por percentil q | `detectors.py` → `BaseDetector.set_threshold` | `threshold.q`, `threshold.sensitivity_q` |
| 3.2.6 | Silhouette y Davies–Bouldin | `objectives.py` → `objective_silhouette`, `objective_davies_bouldin` | `optimization.silhouette_sample`; semilla 0 (fija) |
| 3.2.6 | Error de reconstrucción | `objectives.py` → `objective_reconstruction` | — |
| 3.2.6 | Estabilidad | `objectives.py` → `objective_stability` | `objectives.stability_replicas`, `stability_noise_sigma`; semilla 42 (fija) |
| 3.2.6 | Consistencia | `objectives.py` → `objective_consistency` | `optimization.consistency_boot`, `consistency_frac`; semilla 42 (fija) |
| 3.2.6 | Funciones por detector (Tabla 3.6) | `optimize.py` → `main` | `detectors` |
| 3.2.7 | Línea base | `optimize.py` → `build_baseline` | valores por defecto en `AutoencoderDetector.__init__` |
| 3.2.7 | Búsqueda TPE, sin poda temprana | `optimize.py` → `_run_trials` | `optimization.n_trials`, `optimization.seeds` |
| 3.2.7 | Oráculo (q fijo y q libre) | `optimize.py` → `run_study`, `oracle_score` | `optimization.oracle`, `threshold.oracle_q_range` |
| 3.2.8 | Etapa 1: límites en el PC | `optimize.py` → `run_study`, `measure_latency_ms` | `edge.max_latency_ms`, `dev_slowdown_factor`, `max_model_kb`, `optimization.latency_samples` |
| 3.2.8 | Etapa 2: medición en la Raspberry Pi | `edge_benchmark.py` → `worker`, `meets` | `edge.n_inference_runs`, `n_warmup_runs`, `flow_rate_per_s`, `max_cpu_percent`, `max_ram_mb`, `latency_stat`; semilla 42 (fija) |
| 3.2.8 | Elección de la primera top-k que cumple | `evaluate.py` → `select_entries` | `optimization.top_k` |
| 3.2.8 | Frente de Pareto MCC/latencia | `evaluate.py` → `pareto_front` | — |
| 3.2.9 | Métricas de detección y MCC_π | `common.py` → `supervised_metrics`, `mcc_from_rates`, `ranking_metrics` | `split.report_prevalences` |
| 3.2.9 | Δ, error estándar, distancia al oráculo e hipótesis | `evaluate.py` → `summarize` | — |
| 3.2.10 | Diagnóstico de las funciones objetivo | `diagnose.py` → `main` | `--n 40` |
| 3.2.11 | Versiones de bibliotecas | `optimize.py` → `write_env` | — |
| 3.2.11 | Pruebas anti-fuga (datos sintéticos) | `tests/smoke_test.py` | — |

## 7. Limitaciones 
1. El preprocessor del profesor ajusta su MinMax y su imputación con un `train` que **incluye
   ataques** (solo mínimos, máximos y medias de las columnas numéricas). Aquí, después, se vuelve a
   escalar usando solo el train normal.
2. La selección de variables sin etiquetas (varianza/frecuencia y correlación) puede descartar
   columnas que sí distinguen ataques pero que son raras en el tráfico normal. Es el precio de no
   usar etiquetas.
3. La latencia medida es solo la del **detector**: no incluye el preprocesamiento (one-hot,
   escalado) ni la extracción de flujos en la Pi.
4. La latencia y la CPU de la Pi cambian con la temperatura y la frecuencia del procesador. Para
   tener cifras estables, repite la medición y revisa que el governor esté en `performance`.
5. El flujo completo (preprocesar, splits, optimizar, evaluar y reporte) ya se corrió en el PC con
   los datos reales y con Optuna. Como TensorFlow no está instalado, el autoencoder se entrenó con
   scikit-learn. La medición en la Raspberry Pi todavía está pendiente.

