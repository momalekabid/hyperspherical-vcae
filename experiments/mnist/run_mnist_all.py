#!/usr/bin/env python
"""
Run all three MNIST-VAE variants (Clifford, Power-spherical, vMF / Normal baseline)
in sequence, harvest their CSV results and plot them together.

Assumptions
-----------
• Each experiment script writes a CSV called
      clifford_vae_results.csv
      pws_vae_results.csv
      vmf_vae_results.csv
  either in   experiments/   or   experiments/results/.
• The CSV cell values are of the form  '44.3±2.1'  (mean ± std) or just floats.
• You want to plot the mean accuracies (ignoring the ±std) obtained with
  latent-dim d = 10 for the three evaluation set sizes 100, 600, 1000.
  Feel free to adapt TARGET_D or N_SAMPLES below.
"""

from __future__ import annotations
import subprocess
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

################################################################################
# Configuration
################################################################################
ROOT = Path(__file__).parent           # …/experiments
TARGET_D = 10                          # latent dimension to visualise
N_SAMPLES = [100, 600, 1000]           # the three evaluation set sizes
SCRIPTS = [
    ("Clifford-VAE",      "mnist_clifford.py",      "clifford_vae_results.csv"),
    ("Power-Spherical",   "mnist_powerspherical.py","pws_vae_results.csv"),
    ("vMF-/Normal base",  "mnist_base.py",          "vmf_vae_results.csv"),
]

################################################################################
# Little helpers
################################################################################
def run_script(script: Path) -> None:
    """Run an experiment script as a blocking subprocess."""
    print(f"\n▶ Running: {script}")
    subprocess.run(["python", str(script)], check=True)

def find_csv(csv_name: str) -> Path:
    """Return the Path where *csv_name* was written."""
    for location in (ROOT, ROOT / "results"):
        p = location / csv_name
        if p.exists():
            return p
    raise FileNotFoundError(csv_name)

_re_float = re.compile(r"([0-9]*\.?[0-9]+)")
def parse_cell(cell: str | float | int) -> float:
    """Extract the *mean* part from '44.3±2.1' or return numeric unchanged."""
    if isinstance(cell, (float, int)):
        return float(cell)
    if isinstance(cell, str):
        m = _re_float.match(cell)
        if m:
            return float(m.group(1))
    raise ValueError(f"Cannot parse cell value: {cell}")

################################################################################
# Harvest → CSV & plot
################################################################################
def main() -> None:
    # 1. Launch all three experiment scripts
    for _, script, _ in SCRIPTS:
        run_script(ROOT / script)

    # 2. Collect accuracies into dict {method: [acc_100, acc_600, acc_1000]}
    combined: dict[str, list[float]] = {}
    for label, _, csv_file in SCRIPTS:
        csv_path = find_csv(csv_file)
        df = pd.read_csv(csv_path, index_col=0)
        # Deal with plain or MultiIndex columns transparently
        if any(isinstance(c, str) and "(" in c for c in df.columns):
            # Flatten columns like "('PWS-VAE', '100')" -> "PWS-VAE_100"
            df.columns = [str(c).strip("()").replace("'", "").replace(", ", "_")
                          for c in df.columns]
        # Pick the correct row (latent dim) and slice out the three n_samples
        vals = []
        for n in N_SAMPLES:
            # Try a variety of possible column names
            for key in (str(n),               # Clifford CSV
                        f"Clifford-VAE_{n}",  # just in case
                        f"PWS-VAE_{n}",
                        f"N-VAE_{n}",
                        f"VMF-VAE_{n}",
                        f"{label}_{n}".replace(" ", "_")):
                if key in df.columns:
                    cell = df.loc[TARGET_D, key]
                    vals.append(parse_cell(cell))
                    break
            else:
                raise KeyError(f"Could not find accuracy for {label}, n={n}")
        combined[label] = vals
        print(f"✓ parsed {label:18s}: {vals}")

    # 3. Save tidy CSV
    tidy = pd.DataFrame(combined, index=N_SAMPLES)
    tidy.index.name = "n_samples"
    out_csv = ROOT / "mnist_all_results.csv"
    tidy.to_csv(out_csv)
    print(f"\nCombined results written to  {out_csv}")

    # 4. Plot
    plt.figure(figsize=(6,4))
    for label, vals in combined.items():
        plt.plot(N_SAMPLES, vals, marker="o", linewidth=2, label=label)
    plt.xlabel("number of test samples")
    plt.ylabel(f"accuracy (%)  |  latent dim d = {TARGET_D}")
    plt.title("MNIST latent-space k-NN accuracy")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out_png = ROOT / "mnist_all_plot.png"
    plt.savefig(out_png, dpi=200)
    print(f"Plot saved to  {out_png}")

if __name__ == "__main__":
    main() 