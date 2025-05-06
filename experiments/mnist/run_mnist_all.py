from __future__ import annotations
import subprocess
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

################################################################################
# replication of base experiments from https://arxiv.org/pdf/1804.00891, extended to other distriibutions 
################################################################################
ROOT = Path(__file__).parent           # …/experiments
TARGET_D = 10                          # latent dimension to visualise
N_SAMPLES = [100, 600, 1000]           # the three evaluation set sizes
SCRIPTS = [
    ("Clifford-VAE",      "mnist_clifford.py",      "clifford_vae_results.csv"),
    ("Power-Spherical",   "mnist_powerspherical.py","pws_vae_results.csv"),
    ("vMF & normal base experiments",  "mnist_base.py", "vmf_vae_results.csv"),
]
def run_script(script: Path) -> None:
    print(f"\n▶ Running: {script}")
    subprocess.run(["python", str(script)], check=True)

def main() -> None:
    for _, script, _ in SCRIPTS:
        run_script(ROOT / script)
if __name__ == "__main__":
    main() 