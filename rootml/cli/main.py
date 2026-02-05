import typer

from rootml.config import load_export_config
from rootml.export import export_to_parquet

from rootml.config import load_train_config
from rootml.train.run import run_training

from rootml.attach import attach_scores


app = typer.Typer()


@app.command()
def export(
    config: str = typer.Option(..., help="YAML config file"),
    out: str = typer.Option(..., help="Output Parquet file"),
):
    """
    Export ROOT file to Parquet.
    """

    cfg = load_export_config(config)
    export_to_parquet(cfg, out)

@app.command()
def train(
    data: str = typer.Option(..., help="Parquet dataset"),
    config: str = typer.Option(..., help="Training config"),
    out: str = typer.Option(..., help="Output directory"),
):
    """
    Train ML model on exported dataset.
    """

    cfg = load_train_config(config)

    run_training(data, cfg, out)

@app.command()
def attach(
    input_root: str,
    tree: str,
    scores: str,
    out: str,
):
    """
    Attach ML scores back to ROOT file.
    """

    attach_scores(
        input_root,
        tree,
        scores,
        "event",
        "ml_score",
        out,
    )


if __name__ == "__main__":
    app()
