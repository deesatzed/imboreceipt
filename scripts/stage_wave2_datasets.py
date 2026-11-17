"""Stage Wave-2 Kaggle CSVs (via APFS zero-copy clonefile) + OpenML/imblearn benchmarks."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd
from imblearn.datasets import fetch_datasets

from imboreceipt.bench.datasets import DEFAULT_CACHE_DIR, _safe_fetch_openml, file_sha256

KAGGLE_SRC = Path("/Users/o2satz/Downloads/Kaggle_DataSets")
KAGGLE_DST = DEFAULT_CACHE_DIR / "kaggle"


def _clone_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    res = subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True, text=True)
    if res.returncode != 0:
        shutil.copy2(src, dst)


def main() -> None:
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Stage local Kaggle downloads into workspace cache using APFS zero-space clonefiles
    files_to_stage = [
        (KAGGLE_SRC / "BAF" / "Base.csv", KAGGLE_DST / "BAF" / "Base.csv"),
        (KAGGLE_SRC / "BAF" / "Variant I.csv", KAGGLE_DST / "BAF" / "Variant_I.csv"),
        (KAGGLE_SRC / "BAF" / "Variant II.csv", KAGGLE_DST / "BAF" / "Variant_II.csv"),
        (
            KAGGLE_SRC / "Home_Credit" / "HC_application_train.csv",
            KAGGLE_DST / "Home_Credit" / "HC_application_train.csv",
        ),
        (
            KAGGLE_SRC / "Home_Credit" / "HC_bureau.csv",
            KAGGLE_DST / "Home_Credit" / "HC_bureau.csv",
        ),
        (
            KAGGLE_SRC / "Home_Credit" / "HC_bureau_balance.csv",
            KAGGLE_DST / "Home_Credit" / "HC_bureau_balance.csv",
        ),
        (
            KAGGLE_SRC / "Home_Credit" / "HC_previous_application.csv",
            KAGGLE_DST / "Home_Credit" / "HC_previous_application.csv",
        ),
        (
            KAGGLE_SRC / "Home_Credit" / "HC_POS_CASH_balance.csv",
            KAGGLE_DST / "Home_Credit" / "HC_POS_CASH_balance.csv",
        ),
        (
            KAGGLE_SRC / "Home_Credit" / "HC_credit_card_balance.csv",
            KAGGLE_DST / "Home_Credit" / "HC_credit_card_balance.csv",
        ),
        (
            KAGGLE_SRC / "Home_Credit" / "HC_installments_payments.csv",
            KAGGLE_DST / "Home_Credit" / "HC_installments_payments.csv",
        ),
        (
            KAGGLE_SRC / "IEEE_FRAUD" / "ieee-fraud-detection" / "train_transaction.csv",
            KAGGLE_DST / "IEEE_FRAUD" / "train_transaction.csv",
        ),
        (
            KAGGLE_SRC / "IEEE_FRAUD" / "ieee-fraud-detection" / "train_identity.csv",
            KAGGLE_DST / "IEEE_FRAUD" / "train_identity.csv",
        ),
    ]

    for src, dst in files_to_stage:
        if not src.exists():
            raise FileNotFoundError(f"Missing expected Kaggle file: {src}")
        _clone_or_copy(src, dst)
        print(f"Staged {dst.relative_to(DEFAULT_CACHE_DIR)} ({dst.stat().st_size / 1e6:.1f} MB)")

    # 2. Fetch & cache Tier-1 protein_homo & webpage from imblearn
    for key in ("protein_homo", "webpage"):
        csv_p = DEFAULT_CACHE_DIR / f"tier1_{key}.csv"
        if not csv_p.exists():
            bunch = fetch_datasets(data_home=str(DEFAULT_CACHE_DIR / "raw_imblearn"))[key]
            df = pd.DataFrame(bunch.data, columns=[f"f{i}" for i in range(bunch.data.shape[1])])
            df["__target__"] = (bunch.target == 1).astype(int)
            df.to_csv(csv_p, index=False)
        print(f"Cached {csv_p.name} (sha256={file_sha256(csv_p)[:12]}...)")

    # 3. Fetch & cache Scania APS Failure at Trucks (OpenML 41138: 76,000 rows, 170 sensors)
    aps_p = DEFAULT_CACHE_DIR / "tier2_scania_aps.csv"
    if not aps_p.exists():
        ds = _safe_fetch_openml(41138, DEFAULT_CACHE_DIR)
        ds.frame.to_csv(aps_p, index=False)
    print(f"Cached {aps_p.name} (sha256={file_sha256(aps_p)[:12]}...)")


if __name__ == "__main__":
    main()
