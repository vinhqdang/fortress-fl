"""
Download N-BaIoT from the UCI repository, unpack it and write the sampled
feature file used by experiments/unlearning_real.py --dataset nbaiot.

Needs ~10 GB of temporary disk and the `unar` tool for the .rar archives
(apt install unar). Run once:

    python experiments/prepare_nbaiot.py --raw /tmp/nbaiot_raw
"""

import argparse
import glob
import os
import subprocess
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fortress_fl.unlearning import data

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="/tmp/nbaiot_raw")
ap.add_argument("--out", default="data/nbaiot/sample.npz")
ap.add_argument("--rows", type=int, default=400, help="rows sampled per CSV file")
a = ap.parse_args()

os.makedirs(a.raw, exist_ok=True)
zpath = os.path.join(a.raw, "nbaiot.zip")
if not os.path.exists(zpath):
    urllib.request.urlretrieve(data.NBAIOT_URL, zpath)
with zipfile.ZipFile(zpath) as z:
    z.extractall(a.raw)
for rar in glob.glob(os.path.join(a.raw, "*", "*.rar")):
    subprocess.run(["unar", "-q", "-f", "-o", os.path.dirname(rar), rar], check=True)
print(data.prepare_nbaiot(a.raw, a.out, rows_per_file=a.rows))
