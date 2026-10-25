"""Sort the MRL Eye Dataset into data/eyes/{open,closed}.

MRL filenames: subject_image_gender_glasses_eyeState_reflections_lighting_sensor.png
eyeState: 0 = closed, 1 = open.   Usage: python scripts/prepare_mrl.py PATH_TO_MRL --limit 15000
"""
import argparse
import random
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--dst", default="data/eyes")
ap.add_argument("--limit", type=int, default=15000, help="max images per class")
a = ap.parse_args()

groups = {"closed": [], "open": []}
for p in Path(a.src).rglob("*.png"):
    parts = p.stem.split("_")
    if len(parts) >= 5 and parts[4] in ("0", "1"):
        groups["open" if parts[4] == "1" else "closed"].append(p)

random.seed(42)
for name, files in groups.items():
    random.shuffle(files)
    out = Path(a.dst) / name
    out.mkdir(parents=True, exist_ok=True)
    for f in files[: a.limit]:
        shutil.copy2(f, out / f.name)
    print(f"{name}: {min(len(files), a.limit)} of {len(files)}")
