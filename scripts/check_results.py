"""Print summary of novel_final experiment results."""
import pandas as pd
import glob
import os

csvs = sorted(glob.glob(os.path.join("results", "novel_final", "*", "seed_42", "progress.csv")))
if not csvs:
    print("No results found yet.")
else:
    print(f"{'Config':<30s} {'Best':>8s} {'Final':>8s}")
    print("-" * 50)
    for c in csvs:
        df = pd.read_csv(c)
        ev = df.dropna(subset=["eval/mean_return"])
        parts = c.replace("\\", "/").split("/")
        name = [p for p in parts if p.startswith("K")][0]
        if len(ev) > 0:
            best = ev["eval/mean_return"].max()
            final = ev["eval/mean_return"].iloc[-1]
            print(f"{name:<30s} {best:>8.1f} {final:>8.1f}")
        else:
            print(f"{name:<30s} {'N/A':>8s} {'N/A':>8s}")
