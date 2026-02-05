import ROOT
import pandas as pd

# Files
root_file = "synthetic.root"
score_file = "outputs/train_run_1/scores.parquet"
out_file = "synthetic_with_scores.root"
tree_name = "Events"

# Load scores
df = pd.read_parquet(score_file)

# Map: event -> score
score_map = dict(zip(df["event"], df["ml_score"]))

# Open input ROOT
fin = ROOT.TFile.Open(root_file)
tree = fin.Get(tree_name)

# Create output file
fout = ROOT.TFile(out_file, "RECREATE")
newtree = tree.CloneTree(0)

# New branch
ml_score = ROOT.std.vector("double")(1)
branch = newtree.Branch("ml_score", ml_score)

# Loop
for entry in tree:
    evt = int(entry.event[0])

    score = score_map.get(evt, -1.0)
    ml_score[0] = score

    newtree.Fill()

# Save
fout.Write()
fout.Close()
fin.Close()

print("Done:", out_file)

