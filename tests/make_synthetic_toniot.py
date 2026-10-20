"""
make_synthetic_toniot.py - genera datos SINTETICOS con el mismo formato que deja el
preprocessor del profesor para Ton-IoT (leido de su codigo: 02_preprocessing.py):

    <datasets_root>/Parquet/Ton-IoT/train_stratified.(parquet|csv)  y  test_stratified.*
    columnas: caracteristicas numericas en [0,1] (MinMax) + one-hot int8
              + `category` (int8: 0 normal, 1 ataque) + `label` (str '0'/'1') + `type` (str)

Sirve para probar el pipeline sin el dataset real. NO son datos de TON_IoT.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

NUM = ["duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts",
       "src_ip_bytes", "dst_pkts", "dst_ip_bytes", "http_request_body_len",
       "http_response_body_len"]
ATTACKS = {"ddos": 0.35, "scanning": 0.25, "password": 0.15, "xss": 0.10, "injection": 0.10,
           "mitm": 0.05}


def make(n_normal=6000, n_attack=2400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def flows(n, kind):
        d = {}
        if kind == "normal":
            d["duration"] = rng.lognormal(0.0, 1.0, n)
            d["src_bytes"] = rng.lognormal(6.0, 1.0, n)
            d["dst_bytes"] = rng.lognormal(7.0, 1.2, n)
            d["src_pkts"] = rng.poisson(8, n) + 1
            d["dst_pkts"] = rng.poisson(10, n) + 1
        elif kind == "ddos":
            d["duration"] = rng.lognormal(-3.0, 0.6, n)
            d["src_bytes"] = rng.lognormal(4.0, 0.5, n)
            d["dst_bytes"] = np.zeros(n)
            d["src_pkts"] = rng.poisson(400, n) + 50
            d["dst_pkts"] = np.zeros(n)
        elif kind == "scanning":
            d["duration"] = rng.lognormal(-4.0, 0.5, n)
            d["src_bytes"] = rng.lognormal(3.5, 0.3, n)
            d["dst_bytes"] = rng.lognormal(3.0, 0.5, n) * (rng.random(n) < 0.3)
            d["src_pkts"] = np.ones(n)
            d["dst_pkts"] = (rng.random(n) < 0.3).astype(float)
        else:                                             # ataques de capa de aplicacion
            d["duration"] = rng.lognormal(0.5, 1.0, n)
            d["src_bytes"] = rng.lognormal(8.0 if kind != "mitm" else 6.5, 1.2, n)
            d["dst_bytes"] = rng.lognormal(7.5, 1.2, n)
            d["src_pkts"] = rng.poisson(20, n) + 1
            d["dst_pkts"] = rng.poisson(15, n) + 1
        d["missed_bytes"] = np.zeros(n)
        d["src_ip_bytes"] = d["src_bytes"] + 40 * d["src_pkts"]
        d["dst_ip_bytes"] = d["dst_bytes"] + 40 * d["dst_pkts"]
        d["http_request_body_len"] = (rng.lognormal(5, 1, n) if kind in ("password", "xss", "injection")
                                      else np.zeros(n))
        d["http_response_body_len"] = rng.lognormal(7, 1, n) * (rng.random(n) < 0.3)
        # one-hot
        tcp = rng.random(n) < (0.9 if kind != "normal" else 0.55)
        d["proto_tcp"], d["proto_udp"] = tcp.astype(np.int8), (~tcp).astype(np.int8)
        http = rng.random(n) < (0.8 if kind in ("password", "xss", "injection", "mitm") else 0.2)
        d["service_http"] = http.astype(np.int8)
        d["service_dns"] = ((~http) & ~tcp & (rng.random(n) < 0.6)).astype(np.int8)
        d["conn_state_S0"] = (rng.random(n) < (0.9 if kind in ("ddos", "scanning") else 0.02)).astype(np.int8)
        d["conn_state_SF"] = (1 - d["conn_state_S0"]).astype(np.int8)
        d["conn_state_REJ"] = (rng.random(n) < (0.5 if kind == "scanning" else 0.01)).astype(np.int8)
        for j in range(30):                              # one-hot dispersos (dns_query, http_uri, ...)
            p = 0.02 + 0.3 * rng.random()
            d[f"cat_{j:02d}"] = (rng.random(n) < p * (3 if (kind != "normal" and j % 7 == 0) else 1)
                                 ).astype(np.int8)
        d["const_col"] = np.zeros(n, dtype=np.int8)
        d["rare_col"] = (rng.random(n) < 0.00005).astype(np.int8)
        d["dup_src_bytes"] = d["src_bytes"] * 0.999 + 1.0   # casi identica a src_bytes (correlada)
        out = pd.DataFrame(d)
        out["type"] = kind
        return out

    parts = [flows(n_normal, "normal")]
    shares = np.array(list(ATTACKS.values()))
    for (k, _), n in zip(ATTACKS.items(), np.round(shares * n_attack).astype(int)):
        parts.append(flows(max(int(n), 1), k))
    df = pd.concat(parts, ignore_index=True)
    # MinMax global como hace el preprocessor (colas largas -> la mayoria de valores ~ 0)
    for c in NUM + ["dup_src_bytes"]:
        v = df[c].to_numpy(dtype=float)
        df[c] = (v - v.min()) / (v.max() - v.min() + 1e-12)
    df["category"] = (df["type"] != "normal").astype(np.int8)
    df["label"] = df["category"].astype(str)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write(datasets_root: Path, name="Ton-IoT", seed=0, n_normal=6000, n_attack=2400) -> None:
    df = make(n_normal, n_attack, seed)
    cut = int(0.8 * len(df))
    base = Path(datasets_root) / "Parquet" / name
    base.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        ext = "parquet"
    except Exception:
        ext = "csv"
    for part, d in (("train_stratified", df.iloc[:cut]), ("test_stratified", df.iloc[cut:])):
        d = d.reset_index(drop=True)
        (d.to_parquet if ext == "parquet" else d.to_csv)(base / f"{part}.{ext}", index=False)


if __name__ == "__main__":
    write(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Datasets"))
