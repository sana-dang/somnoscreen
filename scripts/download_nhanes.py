"""Optional: download the official NHANES 2017-18 files from the CDC and build the 3 input CSVs.

The project was built on CSV copies of NHANES 2017-18 (demographics / examination /
questionnaire). This script recreates equivalent CSVs from the official CDC SAS transport (XPT)
files, so anyone can rebuild the data from the source:

    DEMO_J (demographics), BMX_J (body measures), BPX_J (blood pressure),
    SLQ_J (sleep), PAQ_J (physical activity), SMQ_J (smoking), DPQ_J (PHQ-9), HUQ_J (health)

Run:  python scripts/download_nhanes.py                      (writes to data/raw/2017-2018-cdc/)
      python scripts/download_nhanes.py --out data/raw/2017-2018

Checked on 2026-10-16: the CSVs built by this script decode to exactly the same table (same rows,
same order, same values) as the CSV copies used for the thesis.
"""

import argparse
import io
import urllib.request
from pathlib import Path

import pandas as pd

from somnoscreen.config import ROOT
from somnoscreen.features import nhanes_columns

URL_PATTERNS = [
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{name}.xpt",
    "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/{name}.XPT",
]
FILES = {
    "demographics": ["DEMO_J"],
    "examination": ["BMX_J", "BPX_J"],
    "questionnaire": ["SLQ_J", "PAQ_J", "SMQ_J", "DPQ_J", "HUQ_J"],
}


def download_xpt(name: str, cache_dir: Path) -> pd.DataFrame:
    """Download one XPT file (cached on disk) and read it with pandas.read_sas."""
    path = cache_dir / f"{name}.xpt"
    if not path.exists():
        for pattern in URL_PATTERNS:
            try:
                with urllib.request.urlopen(pattern.format(name=name), timeout=60) as response:
                    path.write_bytes(response.read())
                break
            except OSError as error:
                print(f"  {pattern.format(name=name)} failed: {error}")
        else:
            raise RuntimeError(f"Could not download {name}")
    df = pd.read_sas(io.BytesIO(path.read_bytes()), format="xport")
    # read_sas turns exact zeros into tiny numbers such as 5.4e-79; put the zeros back.
    numeric = df.select_dtypes("number").columns
    df[numeric] = df[numeric].mask(df[numeric].abs() < 1e-10, 0.0)
    return df


def main() -> None:
    """Download, merge per source file (outer join on SEQN) and save the 3 CSVs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "data" / "raw" / "2017-2018-cdc"))
    out_dir = Path(parser.parse_args().out)
    cache_dir = out_dir / "xpt"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for source, names in FILES.items():
        merged = None
        for name in names:
            print(f"Downloading {name} ...")
            part = download_xpt(name, cache_dir)
            merged = part if merged is None else merged.merge(part, on="SEQN", how="outer")
        needed = ["SEQN", *nhanes_columns(source)]
        missing = [c for c in needed if c not in merged.columns]
        if missing:
            raise KeyError(f"{source}: variables not found in CDC files: {missing}")
        merged[needed].to_csv(out_dir / f"{source}.csv", index=False)
        print(f"Saved {out_dir / f'{source}.csv'} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
