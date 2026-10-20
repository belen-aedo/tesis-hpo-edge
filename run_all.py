"""
run_all.py - ejecuta las etapas del proyecto en orden (Windows, Linux o Raspberry).

Etapas (en este orden):
    preprocesar   pasos 00-03 del preprocessor del profesor (CSV -> Parquet)   [PC]
    splits        prepare_splits.py : bloques train / val / test                [PC]
    diagnostico   diagnose.py       : ¿sirven las metricas? (opcional)          [PC]
    optimizar     optimize.py --all : Optuna + oraculo + linea base             [PC]
    borde         edge_benchmark.py : latencia / CPU / RAM                      [RASPBERRY]
    evaluar       evaluate.py       : evaluacion final sobre test_final         [PC]
    reporte       report.py         : tablas LaTeX                              [PC]

Por defecto corre:  preprocesar, splits, optimizar, evaluar, reporte
('borde' se corre en la Raspberry; 'diagnostico' es opcional).

EJEMPLOS
    python run_all.py --rapido                       # prueba corta de todo
    python run_all.py --etapas splits,optimizar
    python run_all.py --etapas borde                 # en la Raspberry
    python run_all.py --etapas evaluar,reporte       # al volver a la PC
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PRE_V2 = ROOT / "preprocessor" / "preprocessor_v2"
CSV_REL = "Datasets/Sources/TON-IoT/Train_Test_datasets/Train_Test_Network_dataset"
DEFAULT = ["preprocesar", "splits", "optimizar", "evaluar", "reporte"]
ALL = ["preprocesar", "splits", "diagnostico", "optimizar", "borde", "evaluar", "reporte"]


def run(cmd: list[str], cwd: Path, env_extra: dict | None = None) -> None:
    env = dict(os.environ, **(env_extra or {}))
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env)
    if r.returncode != 0:
        sys.exit(f"\nFALLO (codigo {r.returncode}) en: {' '.join(str(c) for c in cmd)}")
    print(f"   ok en {time.time() - t0:.1f}s", flush=True)


def etapa_preprocesar(cfg_args: list[str], features_mode: str) -> None:
    if not PRE_V2.exists():
        sys.exit("Falta el preprocessor del profesor en preprocessor/ (vea LEEME en esa carpeta).")
    run([sys.executable, ROOT / "scripts" / "configurar_preprocessor.py"], ROOT)
    # Protege el CSV original: el paso 00 lee 'Train_Test_Labels.csv' si existe y escribe
    # Train_Test_Network.csv (le agrega la columna `category`). Con la copia, nunca lee y
    # escribe el mismo archivo.
    folder = ROOT / CSV_REL
    net, lab = folder / "Train_Test_Network.csv", folder / "Train_Test_Labels.csv"
    if not net.exists() and not lab.exists():
        sys.exit(f"Falta el dataset: copie Train_Test_Network.csv en {folder}")
    if not lab.exists():
        print(f"copia de respaldo del CSV original -> {lab.name}")
        shutil.copy2(net, lab)
    # PYTHONUTF8: el codigo del profesor imprime emojis y lee config.yaml con open() sin
    # codificacion; en Windows (cp1252) eso falla si la salida se redirige o hay tildes.
    env = {"KMP_DUPLICATE_LIB_OK": "True", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    ds = "Ton-IoT"
    pasos = ["00_source_preprocessing/run.py", "01_train_test_split.py",
             "02_preprocessing.py", "03_stratified_subset.py"]
    if features_mode == "selector":
        pasos.append("04_feature_selection.py")
    for p in pasos:
        run([sys.executable, p, ds], PRE_V2, env)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--etapas", default=",".join(DEFAULT),
                    help=f"lista separada por comas de: {', '.join(ALL)}")
    ap.add_argument("--config", default=None, help="yaml que se mezcla sobre config/experiment.yaml")
    ap.add_argument("--rapido", action="store_true", help="optimize con 25 trials y 1 semilla")
    ap.add_argument("--saltar-preprocessor", action="store_true",
                    help="no correr los pasos del profesor (ya existen los Parquet)")
    args = ap.parse_args()

    etapas = [e.strip() for e in args.etapas.split(",") if e.strip()]
    malas = [e for e in etapas if e not in ALL]
    if malas:
        sys.exit(f"Etapas desconocidas: {malas}. Opciones: {ALL}")
    if args.saltar_preprocessor and "preprocesar" in etapas:
        etapas.remove("preprocesar")
    cfg = ["--config", args.config] if args.config else []

    sys.path.insert(0, str(SRC))
    from common import load_config
    features_mode = load_config(args.config)["features"]["mode"]

    py = sys.executable
    for e in etapas:
        print(f"\n{'=' * 70}\n ETAPA: {e}\n{'=' * 70}", flush=True)
        if e == "preprocesar":
            etapa_preprocesar(cfg, features_mode)
        elif e == "splits":
            run([py, SRC / "prepare_splits.py", *cfg], ROOT)
        elif e == "diagnostico":
            run([py, SRC / "diagnose.py", *cfg], ROOT)
        elif e == "optimizar":
            run([py, SRC / "optimize.py", "--all", *(["--fast"] if args.rapido else []), *cfg], ROOT)
        elif e == "borde":
            run([py, SRC / "edge_benchmark.py", *cfg], ROOT)
        elif e == "evaluar":
            run([py, SRC / "evaluate.py", *cfg], ROOT)
        elif e == "reporte":
            run([py, SRC / "report.py", *cfg], ROOT)
    print("\nListo.")


if __name__ == "__main__":
    main()
