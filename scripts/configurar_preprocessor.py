"""
configurar_preprocessor.py - ajusta la configuracion del preprocessor del profesor
para que apunte a esta carpeta. No cambia ningun .py del profesor.

Hace dos cambios (la primera vez guarda una copia de respaldo *.bak_proyecto):

  1) preprocessor/config.yaml, en el entorno `high_performance`:
         ROOT:    <esta carpeta>
         RESULTS: <esta carpeta>/RESULTS
     Asi el preprocessor lee de Datasets/Sources/... y escribe en Datasets/Parquet/...
     dentro del proyecto.

  2) preprocessor/config.json, en el dataset Ton-IoT: le quita la "/" inicial a la
     ruta ("path") del CSV:
         "/Datasets/Sources/TON-IoT/..."   ->   "Datasets/Sources/TON-IoT/..."
     Funciona con cualquier nombre de archivo que tenga ese "path" (por ejemplo
     Train_Test_Network.csv o preprocessed.csv). El motivo: en Windows, el codigo
     del profesor hace os.path.join(ROOT, "/Datasets/...") y, por esa "/" inicial,
     se pierde ROOT y termina buscando el CSV en C:\\Datasets\\... en vez de dentro
     del proyecto. En Linux no pasa.

USO
    python scripts/configurar_preprocessor.py
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "preprocessor"
DEFAULT_REL = "Datasets/Sources/TON-IoT/Train_Test_datasets/Train_Test_Network_dataset/"


def backup(p: Path) -> None:
    b = p.with_name(p.name + ".bak_proyecto")
    if not b.exists():
        shutil.copy2(p, b)


def yaml_quote(s: str) -> str:
    """Escribe el texto como cadena YAML entre comillas dobles y solo con caracteres
    ASCII (las tildes quedan como \\uXXXX). Asi el valor se lee bien aunque el codigo
    del profesor abra el archivo con la codificacion de Windows."""
    out = []
    for ch in s:
        if ch in '"\\':
            out.append("\\" + ch)
        elif ord(ch) < 128:
            out.append(ch)
        elif ord(ch) <= 0xFFFF:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(f"\\U{ord(ch):08x}")
    return '"' + "".join(out) + '"'


def patch_yaml(text: str, root_posix: str) -> tuple[str, bool]:
    """Cambia ROOT y RESULTS, pero solo dentro del bloque `  high_performance:`.
    Respeta los saltos de linea CRLF del archivo."""
    lines = text.splitlines(keepends=True)
    start = next((i for i, ln in enumerate(lines) if re.match(r"^  high_performance:\s*$", ln.rstrip("\r\n"))), None)
    if start is None:
        return text, False
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if re.match(r"^  \S", lines[j]):          # aqui empieza el siguiente entorno (misma sangria)
            end = j
            break
    changed = False
    for k in range(start + 1, end):
        eol = "\r\n" if lines[k].endswith("\r\n") else ("\n" if lines[k].endswith("\n") else "")
        if re.match(r"^    ROOT:", lines[k]):
            lines[k] = f"    ROOT: {yaml_quote(root_posix)}{eol}"
            changed = True
        elif re.match(r"^    RESULTS:", lines[k]):
            lines[k] = f"    RESULTS: {yaml_quote(root_posix + '/RESULTS')}{eol}"
            changed = True
    return "".join(lines), changed


def patch_json(jtext: str) -> tuple[str, str | None, bool]:
    """Le quita la '/' inicial al "path" del dataset Ton-IoT (solo en ese bloque).
    Devuelve (texto, path final, si hubo cambio). El path final es None si no se
    encontro el bloque."""
    m = re.search(r'"Ton-IoT"\s*:\s*\{', jtext)
    if not m:
        return jtext, None, False
    start = m.end()
    nxt = re.search(r'\n {4}"[^"]+"\s*:\s*\{', jtext[start:])      # siguiente dataset
    end = start + nxt.start() if nxt else len(jtext)
    block = jtext[start:end]
    pm = re.search(r'("path"\s*:\s*")([^"]*)(")', block)             # busca "path", no "paths"
    if not pm:
        return jtext, None, False
    old = pm.group(2)
    new = old.lstrip("/\\")
    if new == old:
        return jtext, new, False
    block2 = block[:pm.start(2)] + new + block[pm.end(2):]
    return jtext[:start] + block2 + jtext[end:], new, True


def main() -> int:
    yml, js, v2 = PRE / "config.yaml", PRE / "config.json", PRE / "preprocessor_v2"
    faltan = [str(p.relative_to(ROOT)) for p in (yml, js, v2) if not p.exists()]
    if faltan:
        print("Falta copiar el preprocessor del profesor dentro de la carpeta 'preprocessor/'.\n"
              "Faltan: " + ", ".join(faltan) + "\n"
              "Descomprima preprocessor.zip y copie el CONTENIDO de su carpeta 'preprocessor' "
              "(preprocessor_v2/, config.yaml, config.json, ...) dentro de 'preprocessor/'.")
        return 1

    root_posix = ROOT.as_posix()
    with open(yml, "r", encoding="utf-8", newline="") as fh:       # newline="" para no cambiar los saltos de linea CRLF
        text = fh.read()
    new, changed = patch_yaml(text, root_posix)
    if not changed:
        print("[!] No encontre ROOT/RESULTS en el entorno high_performance de config.yaml; "
              "editelos a mano:  ROOT: \"" + root_posix + "\"")
        return 1
    backup(yml)
    with open(yml, "w", encoding="utf-8", newline="") as fh:
        fh.write(new)
    print(f"config.yaml  -> ROOT = {root_posix}")

    with open(js, "r", encoding="utf-8", newline="") as fh:
        jtext = fh.read()
    new_text, rel, changed = patch_json(jtext)
    if rel is None:
        print('[!] No encontre el "path" del dataset Ton-IoT en config.json; debe quedar SIN "/" '
              'inicial, p. ej. "Datasets/Sources/TON-IoT/Train_Test_datasets/'
              'Train_Test_Network_dataset/preprocessed.csv".')
        return 1
    if changed:
        backup(js)
        with open(js, "w", encoding="utf-8", newline="") as fh:
            fh.write(new_text)
        print(f"config.json  -> path de Ton-IoT sin '/' inicial: {rel}")
    else:
        print(f"config.json  -> ya estaba corregido: {rel}")

    folder = (ROOT / rel).parent
    raw = next((p for p in folder.glob("*.csv") if p.name.lower() == "train_test_network.csv"), None) \
        if folder.exists() else None
    lab = folder / "Train_Test_Labels.csv"
    if raw is not None or lab.exists():
        f = raw or lab
        print(f"\nDataset encontrado: {f}  ({f.stat().st_size / 1e6:.0f} MB)")
    else:
        print(f"\n[!] Falta el dataset. Copie Train_Test_Network.csv en:\n    {folder}\n"
              "(lo descarga de la pagina oficial de TON_IoT de UNSW, carpeta "
              "Train_Test_datasets/Train_Test_Network_dataset).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
