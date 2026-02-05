import ROOT
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import subprocess
import json
import time

from rootml.config import ExportConfig


def _git_hash():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def export_to_parquet(cfg: ExportConfig, out_path: str):

    ROOT.ROOT.EnableImplicitMT()

    print("Initializing RDataFrame...")
    df = ROOT.RDataFrame(cfg.tree, cfg.input_files)

    if cfg.selection:
        print(f"Applying selection: {cfg.selection}")
        df = df.Filter(cfg.selection)

    cols = (
        cfg.features
        + [cfg.label]
        + cfg.event_id
        + ([cfg.weight] if cfg.weight else [])
    )

    print("Exporting columns:", cols)

    arrays = df.AsNumpy(cols)

    pdf = pd.DataFrame(arrays)

    metadata = {
        "rootml_export_time": time.ctime(),
        "git_commit": _git_hash(),
        "config": json.dumps(cfg.__dict__, indent=2),
    }

    print("Writing Parquet with provenance...")

    table = pa.Table.from_pandas(pdf)

    # Attach metadata to schema (PyArrow-compatible way)
    schema = table.schema.with_metadata(
        {k.encode(): v.encode() for k, v in metadata.items()}
    )

    table = table.cast(schema)

    pq.write_table(table, out_path)


    print(f"Done. Wrote {len(pdf)} rows to {out_path}")
