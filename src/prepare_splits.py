"""
prepare_splits.py - Etapa 1: toma la salida del preprocessor del profesor y arma
los bloques del experimento (los 3 bloques del diagrama aprobado).

    train : solo trafico normal, sin etiquetas
    val   : solo trafico normal, sin etiquetas (calibra el umbral y calcula el objetivo)
    test  : normal + ataque, con etiquetas y separado del resto. Se divide en dos
            partes sin filas en comun, manteniendo la proporcion de cada tipo de
            ataque y con semilla fija:
              test_oracle : solo la usa el escenario oraculo (maximiza el MCC)
              test_final  : se abre una sola vez (evaluate.py) y es la misma
                            para todos los escenarios

De donde sale cada bloque
-------------------------
El preprocessor entrega `train_*` (80 %) y `test_*` (20 %), ambos con trafico
normal y ataques. Su escalado MinMax y su imputacion se ajustan solo con su train.
  * train y val salen del trafico normal del train del preprocessor. Los
    detectores aprenden solo lo normal, asi que sus ataques se descartan.
  * test es el test del preprocessor completo. Asi, los ataques de prueba nunca
    se usaron para ajustar el preprocessor.

El resto de la cadena no usa etiquetas
--------------------------------------
  * Seleccion de caracteristicas (features.mode = unsupervised): usa solo el
    train normal (varianza/frecuencia y correlacion). No mira etiquetas.
  * Reescalado (scaling.method = standard): media y desviacion del train normal.
    Las columnas 0/1 (one-hot) se dejan igual.
  Ojo: el paso 04 del profesor (features.mode = selector) usa etiquetas (salvo
  `vif`). Se puede usar, pero hay que declararlo como limitacion.

USO
    python src/prepare_splits.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (datasets_dir, find_table, io_ext, load_config, outputs_dir, read_table,
                    save_json, set_seeds, sha256_of, splits_dir, write_table)

HINT = {
    "stratified": "paso 03 (03_stratified_subset.py)",
    "processed": "paso 02 (02_preprocessing.py)",
    "selector": "paso 04 (04_feature_selection.py)",
}


# --------------------------------------------------------------------------- #
# Lectura de la salida del preprocessor
# --------------------------------------------------------------------------- #
def input_bases(cfg) -> tuple[Path, Path, str]:
    base = datasets_dir(cfg) / "Parquet" / cfg["dataset"]["name"]
    mode = cfg["features"]["mode"]
    if mode == "selector":
        sel = cfg["features"]["selector"]
        return (base / sel / "train" / "best_features", base / sel / "test" / "best_features",
                "selector")
    v = cfg["input"]["variant"]
    if v not in ("stratified", "processed"):
        raise SystemExit(f"input.variant debe ser 'stratified' o 'processed' (es '{v}').")
    return base / f"train_{v}", base / f"test_{v}", v


def read_inputs(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr_base, te_base, kind = input_bases(cfg)
    try:
        tr_path, te_path = find_table(tr_base), find_table(te_base)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"\nNo encuentro la salida del preprocessor:\n  {exc}\n"
            f"Falta ejecutar el {HINT[kind]} para '{cfg['dataset']['name']}'.\n"
            f"Carpeta buscada: {tr_base.parent}\n"
            "Revise tambien paths.datasets_root en config/experiment.yaml.")
    print(f"train del preprocessor: {tr_path}\ntest  del preprocessor: {te_path}")
    return read_table(tr_path), read_table(te_path)


def build_target(df: pd.DataFrame, cfg) -> np.ndarray:
    """1 = ataque, 0 = normal."""
    col, nv = cfg["dataset"]["label_col"], cfg["dataset"]["normal_value"]
    if col not in df.columns:
        raise SystemExit(f"La columna de etiqueta '{col}' no esta en los datos. "
                         f"Columnas (primeras 15): {list(df.columns)[:15]}")
    raw = df[col]
    if pd.api.types.is_numeric_dtype(raw):
        try:
            return (raw.to_numpy(dtype=np.float64) != float(nv)).astype(int)
        except (TypeError, ValueError):
            pass
    return (raw.astype(str).str.strip().str.lower() != str(nv).strip().lower()).astype(int)


def attack_types(df: pd.DataFrame, cfg, y: np.ndarray) -> np.ndarray:
    col = cfg["dataset"].get("attack_type_col")
    if col and col in df.columns:
        t = df[col].astype(str).str.strip().to_numpy()
        return np.where(y == 1, t, "normal").astype(object)
    return np.where(y == 1, "ataque", "normal").astype(object)


def feature_frame(df: pd.DataFrame, cfg) -> pd.DataFrame:
    drop = set(cfg["dataset"]["drop_cols"]) | {cfg["dataset"]["label_col"]}
    if cfg["dataset"].get("attack_type_col"):
        drop.add(cfg["dataset"]["attack_type_col"])
    X = df.drop(columns=[c for c in df.columns if c in drop])
    X = X.select_dtypes(include=[np.number, "bool"]).astype(np.float32)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


# --------------------------------------------------------------------------- #
# Seleccion de caracteristicas y escalado SIN etiquetas (solo train normal)
# --------------------------------------------------------------------------- #
def binary_mask(A: np.ndarray, chunk: int = 256) -> np.ndarray:
    """True en las columnas cuyos valores son solo 0 y 1."""
    out = np.empty(A.shape[1], dtype=bool)
    for s in range(0, A.shape[1], chunk):
        blk = A[:, s:s + chunk]
        out[s:s + chunk] = ((blk == 0) | (blk == 1)).all(axis=0)
    return out


def select_unsupervised(Xtr: pd.DataFrame, cfg, seed: int) -> tuple[list[str], dict]:
    """
    Reduce el numero de columnas mirando solo el train normal:
      1. saca las columnas constantes y las columnas 0/1 que casi no cambian (su
         valor menos comun aparece en menos de features.min_minority_freq del
         trafico normal);
      2. las ordena: primero las numericas y despues las 0/1, de mas a menos
         variables (segun p(1-p));
      3. filtro de correlacion: si dos columnas tienen |r| >= max_abs_corr, se
         queda la que va primero. Se detiene al llegar a features.max_features.
    """
    f = cfg["features"]
    A = Xtr.to_numpy(dtype=np.float32)
    cols = np.array(Xtr.columns)
    is_bin = binary_mask(A)
    p = A.mean(axis=0)
    constant = A.max(axis=0) == A.min(axis=0)
    rare = is_bin & (np.minimum(p, 1 - p) < float(f["min_minority_freq"]))
    ok = ~constant & ~rare
    num_idx = np.flatnonzero(ok & ~is_bin)
    bin_idx = np.flatnonzero(ok & is_bin)
    bin_idx = bin_idx[np.argsort(-(p[bin_idx] * (1 - p[bin_idx])), kind="stable")]
    order = np.concatenate([num_idx, bin_idx])
    stats = {"columnas_entrada": int(A.shape[1]), "constantes": int(constant.sum()),
             "binarias_raras": int(rare.sum()), "numericas_candidatas": int(len(num_idx)),
             "binarias_candidatas": int(len(bin_idx))}
    if len(order) == 0:
        raise SystemExit("Tras los filtros no queda ninguna caracteristica: "
                         "revise features.min_minority_freq o los datos de entrada.")

    rng = np.random.default_rng(seed)
    rows = rng.choice(len(A), size=min(len(A), 20000), replace=False)
    S = A[rows][:, order].astype(np.float64)
    S -= S.mean(axis=0)
    sd = S.std(axis=0)
    sd[sd < 1e-12] = np.inf                       # columna constante en la submuestra: r = 0
    S /= sd
    C = (S.T @ S) / len(S)

    kept_pos: list[int] = []
    n_corr = 0
    for i in range(len(order)):
        if len(kept_pos) >= int(f["max_features"]):
            break
        if kept_pos and np.max(np.abs(C[i, kept_pos])) >= float(f["max_abs_corr"]):
            n_corr += 1
            continue
        kept_pos.append(i)
    kept = order[kept_pos]
    stats.update({"descartadas_por_correlacion": n_corr, "conservadas": int(len(kept)),
                  "conservadas_numericas": int(np.isin(kept, num_idx).sum()),
                  "conservadas_binarias": int(np.isin(kept, bin_idx).sum())})
    return [str(c) for c in cols[kept]], stats


def fit_scaler(Xtr: pd.DataFrame, cfg) -> dict:
    """Calcula la media y la desviacion con el train normal. Las columnas 0/1 no se
    escalan (se deja media 0 y desviacion 1)."""
    A = Xtr.to_numpy(dtype=np.float64)
    is_bin = binary_mask(A)
    mu, sd = A.mean(axis=0), A.std(axis=0)
    sd[sd < 1e-12] = 1.0
    mu[is_bin], sd[is_bin] = 0.0, 1.0
    if cfg["scaling"]["method"] == "none":
        mu[:], sd[:] = 0.0, 1.0
    return {"columns": list(Xtr.columns), "mean": mu.tolist(), "std": sd.tolist(),
            "binary": is_bin.tolist(), "method": cfg["scaling"]["method"],
            "clip": cfg["scaling"]["clip"]}


def apply_scaler(X: pd.DataFrame, sc: dict) -> pd.DataFrame:
    A = (X.to_numpy(dtype=np.float64) - np.array(sc["mean"])) / np.array(sc["std"])
    if sc["clip"] is not None:
        A = np.clip(A, -float(sc["clip"]), float(sc["clip"]))
    return pd.DataFrame(A.astype(np.float32), columns=sc["columns"])


# --------------------------------------------------------------------------- #
# Particiones
# --------------------------------------------------------------------------- #
def stratified_halves(rng, keys: np.ndarray, frac_first: float) -> tuple[np.ndarray, np.ndarray]:
    """Divide los indices en dos partes manteniendo la proporcion de cada grupo
    (normal y cada tipo de ataque). Si un grupo tiene una sola fila, va a la
    segunda parte (la de evaluacion)."""
    first, second = [], []
    for g in np.unique(keys):
        idx = rng.permutation(np.flatnonzero(keys == g))
        k = int(np.floor(frac_first * len(idx)))
        first.append(idx[:k])
        second.append(idx[k:])
    return rng.permutation(np.concatenate(first)), rng.permutation(np.concatenate(second))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sp = cfg["split"]
    seed = int(sp["seed"])
    set_seeds(seed)
    rng = np.random.default_rng(seed)

    df_tr, df_te = read_inputs(cfg)
    y_tr, y_te = build_target(df_tr, cfg), build_target(df_te, cfg)
    types_te = attack_types(df_te, cfg, y_te)
    X_tr_all, X_te_all = feature_frame(df_tr, cfg), feature_frame(df_te, cfg)
    print(f"\npreprocessor: train {len(df_tr):,} filas (normal {(y_tr == 0).sum():,}, "
          f"ataque {(y_tr == 1).sum():,}) | test {len(df_te):,} filas "
          f"(normal {(y_te == 0).sum():,}, ataque {(y_te == 1).sum():,}) | "
          f"{X_tr_all.shape[1]} columnas numericas")
    if (y_tr == 0).sum() < 100 or (y_te == 0).sum() == 0 or (y_te == 1).sum() == 0:
        raise SystemExit("Faltan filas normales en el train, o una clase vacia en el test: "
                         "revise dataset.label_col / dataset.normal_value.")

    mode = cfg["features"]["mode"]
    if mode == "selector" and cfg["features"]["selector"] != "vif":
        print(f"[!] AVISO: el selector '{cfg['features']['selector']}' del profesor usa etiquetas "
              "(y su train incluye ataques). Declarelo como limitacion o use features.mode: "
              "unsupervised.")

    # ---- train / val: solo normal del train del preprocessor -------------------
    normal = rng.permutation(np.flatnonzero(y_tr == 0))
    n_val = int(round(float(sp["val_frac_of_normal"]) * len(normal)))
    idx_val, idx_train = normal[:n_val], normal[n_val:]
    Xtrain_raw, Xval_raw = X_tr_all.iloc[idx_train], X_tr_all.iloc[idx_val]

    # ---- caracteristicas y escalado: solo con el train normal ------------------
    if mode == "unsupervised":
        feats, fstats = select_unsupervised(Xtrain_raw.reset_index(drop=True), cfg, seed)
    else:
        feats, fstats = list(X_tr_all.columns), {"conservadas": int(X_tr_all.shape[1])}
    missing = [c for c in feats if c not in X_te_all.columns]
    if missing:
        print(f"[!] AVISO: {len(missing)} columnas del train no estan en el test; se rellenan con 0.")
    Xtrain_raw = Xtrain_raw[feats].reset_index(drop=True)
    Xval_raw = Xval_raw[feats].reset_index(drop=True)
    X_te_sel = X_te_all.reindex(columns=feats, fill_value=0.0)
    scaler = fit_scaler(Xtrain_raw, cfg)
    X_train, X_val = apply_scaler(Xtrain_raw, scaler), apply_scaler(Xval_raw, scaler)
    X_test = apply_scaler(X_te_sel.reset_index(drop=True), scaler)
    print(f"caracteristicas: {len(feats)} ({fstats})")

    # ---- test: (opcional) ajustar la proporcion de ataques; luego dividir en oraculo y final
    keep = np.arange(len(X_test))
    pi = sp.get("test_target_prevalence")
    if pi:
        att, nor = np.flatnonzero(y_te == 1), np.flatnonzero(y_te == 0)
        want = int(round(float(pi) / (1 - float(pi)) * len(nor)))
        if want < len(att):
            att = rng.choice(att, size=want, replace=False)
        else:
            print(f"[!] No hay ataques suficientes para prevalencia {pi:.0%}: "
                  f"se usan los {len(att):,} disponibles.")
        keep = np.sort(np.concatenate([nor, att]))
    y_k, t_k = y_te[keep], types_te[keep]
    o_loc, f_loc = stratified_halves(rng, t_k, float(sp["oracle_frac_of_test"]))
    idx_orc, idx_fin = keep[o_loc], keep[f_loc]

    # ---- escritura ----------------------------------------------------------------
    out = splits_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    ext = io_ext()
    write_table(X_train, out / f"train.{ext}")
    write_table(X_val, out / f"val.{ext}")
    for name, idx in (("test_oracle", idx_orc), ("test_final", idx_fin)):
        d = X_test.iloc[idx].reset_index(drop=True)
        d["y_true"] = y_te[idx]
        d["attack_type"] = types_te[idx]
        write_table(d, out / f"{name}.{ext}")
    X_val.head(int(sp["edge_sample_rows"])).to_csv(out / "edge_sample.csv", index=False)
    save_json(scaler, out / "scaler.json")
    save_json({"features": feats, "mode": mode, "stats": fstats}, out / "features.json")

    def block(idx):
        yy = y_te[idx]
        return {"rows": int(len(idx)), "normal": int((yy == 0).sum()),
                "attack": int((yy == 1).sum()), "attack_ratio": round(float(yy.mean()), 4),
                "attack_types": {t: int((types_te[idx] == t).sum())
                                 for t in sorted(set(types_te[idx]) - {"normal"})}}

    manifest = {
        "dataset": cfg["dataset"]["name"], "input_variant": cfg["input"]["variant"],
        "features_mode": mode, "n_features": len(feats), "feature_names": feats,
        "seed": seed, "format": ext,
        "splits": {
            "train": {"rows": int(len(X_train)), "content": "solo normal, sin etiquetas"},
            "val": {"rows": int(len(X_val)), "content": "solo normal, sin etiquetas"},
            "test_oracle": {**block(idx_orc), "use": "solo escenario oraculo"},
            "test_final": {**block(idx_fin), "use": "solo evaluate.py, una vez"},
        },
        "sha256": {n: sha256_of(out / f"{n}.{ext}")
                   for n in ("train", "val", "test_oracle", "test_final")},
        "note": ("test_final esta aislado: ningun script de optimizacion lo lee. "
                 "evaluate.py verifica el hash antes de abrirlo."),
    }
    save_json(manifest, out / "manifest.json")
    save_json(manifest, outputs_dir(cfg) / "logs" / "splits_manifest.json")

    print(f"\nBloques en {out}  (formato {ext})")
    for k, v in manifest["splits"].items():
        extra = f" | ataque {v['attack_ratio']:.1%}" if "attack_ratio" in v else ""
        print(f"  {k:12s}: {v['rows']:>8,} filas{extra}")
    ratio = manifest["splits"]["test_final"]["attack_ratio"]
    if ratio > 0.5:
        print(f"\n[!] AVISO: {ratio:.0%} de test_final es ataque. Con esa prevalencia la Precision "
              "es enganosamente alta y el MCC no representa un despliegue real. Use "
              "split.test_target_prevalence (p. ej. 0.10) o mire MCC_pi05 / MCC_pi10.")
    if manifest["splits"]["test_oracle"]["normal"] < 300:
        print("[!] AVISO: pocos normales en test_oracle: el FPR estimado sera ruidoso.")
    if len(X_val) < 1000:
        print("[!] AVISO: val tiene menos de 1000 filas: umbral y metricas ruidosos. "
              "Considere input.variant: processed.")


if __name__ == "__main__":
    main()
