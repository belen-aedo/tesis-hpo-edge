"""
common.py - funciones que usan todas las etapas.

Aqui estan la lectura de la configuracion, las rutas, la lectura y escritura de
tablas, las semillas, la division de `val` en calibracion y objetivo, y las
metricas de evaluacion.

Las metricas supervisadas (las que usan etiquetas) solo las llaman el oraculo
(optimize.py, con test_oracle), diagnose.py y evaluate.py (con test_final).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "experiment.yaml"


# --------------------------------------------------------------------------- #
# Configuracion
# --------------------------------------------------------------------------- #
def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Lee config/experiment.yaml. Si se pasa otro yaml en `path`, sus valores
    reemplazan a los del archivo base."""
    with open(DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if path and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path, "r", encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    cfg["_config_path"] = str(path or DEFAULT_CONFIG)
    return cfg


def _resolve(p: str | Path) -> Path:
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else REPO_ROOT / p


def datasets_dir(cfg) -> Path:
    return _resolve(cfg["paths"]["datasets_root"])


def splits_dir(cfg) -> Path:
    return _resolve(cfg["paths"]["splits_dir"])


def outputs_dir(cfg) -> Path:
    return _resolve(cfg["paths"]["outputs_dir"])


def models_dir(cfg) -> Path:
    return outputs_dir(cfg) / "models"


def results_dir(cfg) -> Path:
    return outputs_dir(cfg) / "results"


# --------------------------------------------------------------------------- #
# Entrada / salida
# --------------------------------------------------------------------------- #
def parquet_available() -> bool:
    for mod in ("pyarrow", "fastparquet"):
        try:
            __import__(mod)
            return True
        except Exception:
            continue
    return False


def io_ext() -> str:
    """Formato para guardar los bloques: parquet si hay una libreria que lo lea; si no, csv."""
    return "parquet" if parquet_available() else "csv"


def find_table(base: Path) -> Path:
    """Recibe una ruta sin extension y devuelve el .parquet si existe; si no, el .csv."""
    for ext in ("parquet", "csv"):
        p = Path(f"{base}.{ext}")
        if p.exists():
            return p
    raise FileNotFoundError(f"No se encontro {base}.parquet ni {base}.csv")


def read_table(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontro {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".csv", ".gz"):
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Formato no soportado: {path.suffix}")


def write_table(df: pd.DataFrame, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def sha256_of(path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj: Any, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# Particiones
# --------------------------------------------------------------------------- #
NON_FEATURE = ("y_true", "attack_type")


def split_path(cfg, name: str) -> Path:
    return find_table(splits_dir(cfg) / name)


def load_labeled(cfg, name: str):
    """Devuelve (X, y, tipos de ataque). En train y val, y y los tipos son None,
    porque esos bloques no tienen etiquetas."""
    df = read_table(split_path(cfg, name))
    y = df["y_true"].to_numpy() if "y_true" in df.columns else None
    types = df["attack_type"].astype(str).to_numpy() if "attack_type" in df.columns else None
    X = df.drop(columns=[c for c in NON_FEATURE if c in df.columns]).to_numpy(dtype=np.float32)
    return X, y, types


def load_split(cfg, name: str):
    """Devuelve (X, y). Solo traen etiquetas test_oracle (lo usa el oraculo) y
    test_final (lo usa evaluate.py)."""
    X, y, _ = load_labeled(cfg, name)
    return X, y


def split_val(X_val: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    """
    Parte `val` (solo trafico normal, sin etiquetas) en dos mitades sin filas en
    comun, que siempre salen iguales:
        calibracion: sirve para fijar el umbral (percentil q)
        objetivo:    ahi se calculan las funciones objetivo
    Asi el umbral y la metrica no se calculan con las mismas filas.
    """
    rng = np.random.default_rng(int(cfg["split"]["seed"]))
    idx = rng.permutation(len(X_val))
    n_cal = max(int(round(cfg["threshold"]["calib_frac"] * len(X_val))), 1)
    return X_val[idx[:n_cal]], X_val[idx[n_cal:]]


def subsample(X: np.ndarray, max_rows, seed: int = 42) -> np.ndarray:
    if max_rows is None or len(X) <= max_rows:
        return X
    rng = np.random.default_rng(seed)
    return X[rng.choice(len(X), size=int(max_rows), replace=False)]


def train_for(cfg, detector: str, X_train: np.ndarray) -> np.ndarray:
    """Filas de entrenamiento que usa cada detector. A LOF se le pasan menos, porque
    guarda todo el train en memoria."""
    opt = cfg["optimization"]
    X = subsample(X_train, opt.get("max_train_rows"))
    if detector == "lof":
        X = subsample(X, opt.get("max_train_rows_lof"))
    return X


# --------------------------------------------------------------------------- #
# Metricas supervisadas (oraculo, diagnostico y evaluacion final)
# --------------------------------------------------------------------------- #
def mcc_from_rates(tpr: float, fpr: float, prevalence: float) -> float:
    """MCC que se obtendria si la proporcion de ataques fuera `prevalence`, calculado
    con el TPR y el FPR. Como TPR y FPR no dependen de cuantos ataques haya, sirve
    para reportar el MCC en un escenario con pocos ataques sin tener que botar datos."""
    p, n = prevalence, 1.0 - prevalence
    tp, fn, fp, tn = p * tpr, p * (1 - tpr), n * fpr, n * (1 - fpr)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def supervised_metrics(y_true: np.ndarray, y_pred: np.ndarray, prevalences=()) -> Dict[str, float]:
    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                                 matthews_corrcoef, precision_score, recall_score)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    out = {
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "BACC": float(balanced_accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "FPR": float(fpr),
        "FNR": float(1 - tpr) if (fn + tp) else 0.0,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }
    for pi in prevalences:
        out[f"MCC_pi{int(round(pi * 100)):02d}"] = mcc_from_rates(tpr, fpr, pi)
    return out


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Metricas que no dependen del umbral (AUROC y AUPRC)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    if len(np.unique(y_true)) < 2:
        return {"AUROC": float("nan"), "AUPRC": float("nan")}
    return {"AUROC": float(roc_auc_score(y_true, scores)),
            "AUPRC": float(average_precision_score(y_true, scores))}
