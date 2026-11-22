"""Extrae PlantDoc desde el clon git cuando el checkout falla en Windows.

Algunos archivos de PlantDoc tienen caracteres inválidos en NTFS (p. ej. '?').
Este script lee los blobs con `git cat-file` y los escribe con nombres saneados.

    python scripts/extract_plantdoc.py <ruta_clon_PlantDoc-Dataset>
"""
import re
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1])
tree = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "-z", "HEAD"],
                      capture_output=True, check=True).stdout.decode("utf-8").split("\0")
n = 0
for entry in tree:
    if not entry:
        continue
    meta, path = entry.split("\t", 1)
    if not (path.startswith("train/") or path.startswith("test/")):
        continue
    sha = meta.split()[2]
    parts = path.split("/")
    name = re.sub(r'[<>:"\\|?*]', "_", parts[-1])
    stem, dot, ext = name.rpartition(".")
    if len(name) > 90:  # evita superar MAX_PATH en Windows
        name = f"{stem[:70]}_{sha[:8]}.{ext}"
    parts[-1] = name
    dst = repo.joinpath(*parts)
    if dst.exists():
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    blob = subprocess.run(["git", "-C", str(repo), "cat-file", "blob", sha],
                          capture_output=True, check=True).stdout
    dst.write_bytes(blob)
    n += 1
print(f"Archivos escritos: {n}")
