#!/usr/bin/env python3
"""Download official AI4I 2020 CSV via kagglehub (requires Kaggle API token if prompted)."""
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print("Install kagglehub: pip install kagglehub")
    raise

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download("stephanmatzka/predictive-maintenance-dataset-ai4i-2020")
    src = Path(path)
    csvs = list(src.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV in {src}")
    target = DATA / "ai4i2020.csv"
    # Prefer file named like ai4i2020
    preferred = [c for c in csvs if "ai4i" in c.name.lower()]
    chosen = preferred[0] if preferred else csvs[0]
    target.write_bytes(chosen.read_bytes())
    print(f"Copied {chosen} -> {target}")


if __name__ == "__main__":
    main()
