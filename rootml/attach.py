import ROOT
import pandas as pd


def attach_scores(
    input_root,
    tree_name,
    scores_path,
    event_col,
    score_col,
    output_root,
):

    # Load scores
    scores = pd.read_parquet(scores_path)

    # Build Python lookup
    score_map = dict(
        zip(scores[event_col], scores[score_col])
    )

    ROOT.ROOT.DisableImplicitMT()

    df = ROOT.RDataFrame(tree_name, input_root)

    # Python lookup function
    def lookup(event):
        return float(score_map.get(int(event), -1.0))

    # Attach via Python callable
    df2 = df.Define(
        "ml_score",
        lookup,
        [event_col]
    )

    df2.Snapshot(tree_name, output_root)

    print(f"Saved new ROOT file: {output_root}")
