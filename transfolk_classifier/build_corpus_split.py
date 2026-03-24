import os
import pandas as pd
from sklearn.model_selection import train_test_split


def build_split_metadata(prof_dir: str, reli_dir: str, test_size: float = 0.20, random_state: int = 42):
    rows = []

    for fname in os.listdir(prof_dir):
        if fname.lower().endswith((".xml", ".musicxml", ".mxl")):
            rows.append({
                "__file": fname,
                "__path": os.path.join(prof_dir, fname),
                "label_name": "Profano",
                "label_id": 1,
            })

    for fname in os.listdir(reli_dir):
        if fname.lower().endswith((".xml", ".musicxml", ".mxl")):
            rows.append({
                "__file": fname,
                "__path": os.path.join(reli_dir, fname),
                "label_name": "Religioso",
                "label_id": 0,
            })

    df = pd.DataFrame(rows)

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label_id"]
    )

    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    return df