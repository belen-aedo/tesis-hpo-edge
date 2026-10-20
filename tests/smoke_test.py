"""
smoke_test.py - prueba de humo de punta a punta con datos SINTETICOS (no son TON_IoT).

Comprueba que el pipeline corre (splits -> diagnose -> optimize -> edge_benchmark ->
evaluate -> report) y que NO hay fuga de etiquetas:
  * train y val no tienen etiquetas ni ataques;
  * optimize (escenarios no supervisados) corre con test_oracle y test_final OCULTOS;
  * el oraculo corre con test_final oculto;
  * si el hash de test_final cambia, evaluate.py se niega a correr;
  * un edge_benchmark.csv de version anterior no hace caer a evaluate.py.
Si Optuna no esta instalado usa busqueda aleatoria (ALLOW_RANDOM_FALLBACK=1): eso
valida la logica, NO reemplaza una corrida real con Optuna.

    python tests/smoke_test.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "tests"))
import make_synthetic_toniot as syn   # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="smoke_"))
ds_root = tmp / "Datasets"
syn.write(ds_root)

cfg = {
    "paths": {"datasets_root": str(ds_root), "splits_dir": str(tmp / "splits"),
              "outputs_dir": str(tmp / "outputs")},
    "input": {"variant": "stratified"},          # los datos sinteticos se generan solo en esta variante
    "optimization": {"n_trials": 6, "seeds": [42, 43], "top_k": 3, "latency_samples": 10},
    "edge": {"n_inference_runs": 60, "n_warmup_runs": 5},
    "autoencoder": {"backend": "sklearn"},
    "detectors": {"iforest": ["silhouette", "davies_bouldin"], "lof": ["silhouette"],
                  "autoencoder": ["reconstruction", "stability", "consistency"]},
}
cfgp = tmp / "prueba.yaml"
cfgp.write_text(yaml.safe_dump(cfg))
env = dict(os.environ, ALLOW_RANDOM_FALLBACK="1")


def run(script, *args, expect_fail=False):
    cmd = [sys.executable, str(SRC / script), "--config", str(cfgp), *args]
    print("\n$ python", script, *args)
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    print(p.stdout[-1500:])
    if expect_fail:
        assert p.returncode != 0, f"{script} debia fallar"
        return p
    if p.returncode != 0:
        print(p.stderr[-2500:])
        sys.exit(f"FALLO en {script}")
    return p


def check(cond, msg):
    print(("  OK  " if cond else "  MAL ") + msg)
    if not cond:
        sys.exit("FALLO: " + msg)


# ---- 1) particiones -------------------------------------------------------------
run("prepare_splits.py")
sp = tmp / "splits"
ext = "parquet" if (sp / "train.parquet").exists() else "csv"
man = json.loads((sp / "manifest.json").read_text())
tr, va = pd.read_csv(sp / "train.csv") if ext == "csv" else pd.read_parquet(sp / "train.parquet"), \
    pd.read_csv(sp / "val.csv") if ext == "csv" else pd.read_parquet(sp / "val.parquet")
check("y_true" not in tr.columns and "y_true" not in va.columns, "train/val sin etiquetas")
check(man["n_features"] <= 40 and man["n_features"] > 10, f"{man['n_features']} caracteristicas (<=40)")
feats = man["feature_names"]
check("const_col" not in feats and "rare_col" not in feats, "columna constante y columna rara eliminadas")
check(not ({"src_bytes", "dup_src_bytes"} <= set(feats)), "una de dos columnas correlacionadas eliminada")
check(all(c in feats for c in ["duration", "src_bytes", "dst_bytes"]), "las numericas se conservan")
check(abs(tr[feats[0]].mean()) < 0.05, "train reescalado con sus propios estadisticos")
n_tot_normal = None
for b in ("test_oracle", "test_final"):
    check(set(man["splits"][b]["attack_types"]) == set(syn.ATTACKS), f"{b}: estan todos los tipos de ataque")
tf_ = man["splits"]["test_final"]
check(0.2 < tf_["attack_ratio"] < 0.45, f"prevalencia natural del test: {tf_['attack_ratio']:.1%}")
check(man["splits"]["train"]["rows"] > man["splits"]["val"]["rows"] > 100, "tamanos train > val")
check((sp / "edge_sample.csv").exists(), "edge_sample.csv creado")

# ---- 2) optimize sin ver etiquetas ------------------------------------------------
hidden = tmp / "hidden"
hidden.mkdir()


def hide(*names):
    for n in names:
        shutil.move(sp / f"{n}.{ext}", hidden / f"{n}.{ext}")


def show(*names):
    for n in names:
        shutil.move(hidden / f"{n}.{ext}", sp / f"{n}.{ext}")


hide("test_oracle", "test_final")
run("optimize.py", "--all", "--no-oracle", "--trials", "6", "--seeds", "42", "43")
show("test_oracle")                       # el oraculo solo necesita test_oracle
run("optimize.py", "--detector", "iforest", "--metric", "oracle", "--trials", "6", "--seeds", "42", "43")
run("optimize.py", "--detector", "lof", "--metric", "oracle_q", "--trials", "6", "--seeds", "42", "43")
run("diagnose.py", "--n", "8")
show("test_final")
print("\n>>> LEAK CHECK OK: optimize no necesito test_final ni (salvo oraculo) test_oracle")
check((tmp / "outputs" / "models" / "env.json").exists(), "env.json con versiones de librerias")

# ---- 3) borde, evaluacion, reporte --------------------------------------------------
run("edge_benchmark.py")
edge = pd.read_csv(tmp / "outputs" / "results" / "edge_benchmark.csv")
check({"rank", "cumple_ram", "cumple_cpu", "cumple_latencia"} <= set(edge.columns), "edge_benchmark con esquema nuevo")
run("evaluate.py")
res = tmp / "outputs" / "results"
for f in ("final_evaluation.csv", "selected_models.csv", "balance.csv", "recall_por_ataque.csv",
          "pareto_mcc_latencia.csv", "hypothesis_check.json", "resumen.md"):
    check((res / f).exists(), f"salida {f}")
fe = pd.read_csv(res / "final_evaluation.csv")
check({"linea_base", "optimizada", "oraculo", "linea_base_nativa"} <= set(fe.config), "escenarios completos")
check(fe.loc[fe.config != "linea_base_nativa", "AUROC"].between(0, 1).all(), "AUROC en [0, 1]")
run("report.py")
check("[Pendiente]" not in (tmp / "outputs" / "tables" / "tab_mock_balance.tex").read_text(), "tabla de balance con datos")

# ---- 4) robustez --------------------------------------------------------------------
old = edge.drop(columns=["rank", "cumple_ram"])
old.to_csv(res / "edge_benchmark.csv", index=False)               # esquema viejo
p = run("evaluate.py")
check("version anterior" in p.stdout, "evaluate ignora un edge_benchmark.csv antiguo sin caerse")
run("report.py")
pd.DataFrame({"x": [1]}).to_csv(sp / f"test_final.csv", index=False) if ext == "csv" else None
if ext == "csv":
    p = run("evaluate.py", expect_fail=True)
    check("NO coincide" in (p.stdout + p.stderr), "evaluate rechaza un test_final modificado (hash)")

# ---- 5) modo selector y prevalencia objetivo ------------------------------------------
cfg2 = dict(cfg, features={"mode": "selector", "selector": "vif"},
            split={"test_target_prevalence": 0.10},
            paths=dict(cfg["paths"], splits_dir=str(tmp / "splits2")))
base = ds_root / "Parquet" / "Ton-IoT"
for part in ("train", "test"):
    src_df = pd.read_csv(base / f"{part}_stratified.csv") if (base / f"{part}_stratified.csv").exists() \
        else pd.read_parquet(base / f"{part}_stratified.parquet")
    keep = ["duration", "src_bytes", "dst_bytes", "proto_tcp", "service_http", "category", "label", "type"]
    d = base / "vif" / part
    d.mkdir(parents=True, exist_ok=True)
    src_df[keep].to_csv(d / "best_features.csv", index=False)
cfgp2 = tmp / "prueba2.yaml"
cfgp2.write_text(yaml.safe_dump(cfg2))
p = subprocess.run([sys.executable, str(SRC / "prepare_splits.py"), "--config", str(cfgp2)],
                   capture_output=True, text=True, cwd=ROOT)
print(p.stdout[-900:])
check(p.returncode == 0, "prepare_splits en modo selector")
m2 = json.loads((tmp / "splits2" / "manifest.json").read_text())
check(m2["n_features"] == 5, "modo selector: 5 caracteristicas")
check(abs(m2["splits"]["test_final"]["attack_ratio"] - 0.10) < 0.03,
      f"prevalencia objetivo 10 %: {m2['splits']['test_final']['attack_ratio']:.1%}")

print("\nSMOKE TEST OK ->", tmp)
