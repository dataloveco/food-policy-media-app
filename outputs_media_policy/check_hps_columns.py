# check_hps_columns.py
import sys, re
from pathlib import Path
import pandas as pd

PATTERNS = [
    r"food\s*scar(c|s)it(y|e)",
    r"food\s*scarce",
    r"food\s*insufficien",
    r"hps.*scarce",
    r"food.*insecure",
    r"insufficient.*food",
    r"food.*hardship",
]

FILES = [
    "hps_foodscarce_W.csv",
    "features_W.csv",
    "model_dataset_W.csv",
    "table1_sample_coverage.csv",
    "table1_sample_coverage_vertical.csv",
    "table2_feature_summary.csv",
    "table2_feature_summary_clean.csv",
]

def candidates(df):
    cols = list(df.columns)
    cands = []
    for c in cols:
        cu = c.lower()
        if any(re.search(p, cu) for p in PATTERNS):
            cands.append(c)
    # rank “rate-like” up
    rate_like = [c for c in cands if c.lower().endswith("rate") or "percent" in c.lower() or cu.endswith("%")]
    return rate_like + [c for c in cands if c not in rate_like]

def show(df, name):
    if df is None:
        print(f"  (missing file)")
        return
    print(f"  columns: {list(df.columns)[:12]}{' ...' if len(df.columns)>12 else ''}")
    cands = candidates(df)
    if cands:
        print(f"  candidate HPS columns: {cands[:8]}")
        head = df[[cands[0]]].head(5)
        print("  sample values (first 5):")
        print(head.to_string(index=False))
    else:
        print("  (no HPS-like columns found)")

def read(p):
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"  error reading {p.name}: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_hps_columns.py /path/to/outputs_media_policy")
        sys.exit(1)
    outdir = Path(sys.argv[1])
    print(f"Scanning: {outdir.resolve()}")
    for f in FILES:
        p = outdir / f
        print(f"\n[{f}]")
        df = read(p) if p.exists() else None
        show(df, f)
