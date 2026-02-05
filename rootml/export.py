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

    # IMPORTANT: Disable IMT for chunked processing
    ROOT.ROOT.DisableImplicitMT()

    print("Initializing RDataFrame...")
    df = ROOT.RDataFrame(cfg.tree, cfg.input_files)

    # Apply selection
    if cfg.selection:
        print(f"Applying selection: {cfg.selection}")
        df = df.Filter(cfg.selection)

    # Add entry index for chunking
    df = df.Define("rdf_entry", "rdfentry_")

    cols = (
        cfg.features
        + [cfg.label]
        + cfg.event_id
        + ([cfg.weight] if cfg.weight else [])
    )

    # Include rdf_entry temporarily
    read_cols = ["rdf_entry"] + cols

    print("Exporting columns:", cols)

    # Get total entries
    n = df.Count().GetValue()
    print(f"Total entries: {n}")

    # Provenance metadata
    metadata = {
        "rootml_export_time": time.ctime(),
        "git_commit": _git_hash(),
        "config": json.dumps(cfg.__dict__, indent=2),
    }

    writer = None
    rows_written = 0

    for start in range(0, n, cfg.chunk_size):

        stop = min(start + cfg.chunk_size, n)

        print(f"Processing rows {start} → {stop}")

        # Filter by entry index
        chunk_df = df.Filter(
            f"rdf_entry >= {start} && rdf_entry < {stop}"
        )

        chunk = chunk_df.AsNumpy(read_cols)

        pdf = pd.DataFrame(chunk)

        # Drop helper column
        pdf = pdf.drop(columns=["rdf_entry"])

        table = pa.Table.from_pandas(pdf)

        # Attach metadata on first chunk
        if writer is None:

            schema = table.schema.with_metadata(
                {k.encode(): v.encode() for k, v in metadata.items()}
            )

            table = table.cast(schema)

            writer = pq.ParquetWriter(
                out_path,
                table.schema,
                compression="snappy",
            )

        writer.write_table(table)

        rows_written += len(pdf)

    if writer:
        writer.close()

    print(f"Export complete: {rows_written} rows → {out_path}")
