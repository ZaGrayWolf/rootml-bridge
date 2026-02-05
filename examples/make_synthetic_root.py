import ROOT
import random

f = ROOT.TFile("synthetic.root", "RECREATE")
t = ROOT.TTree("Events", "Synthetic dataset")

run   = ROOT.std.vector("int")(1)
lumi  = ROOT.std.vector("int")(1)
event = ROOT.std.vector("int")(1)

x1 = ROOT.std.vector("float")(1)
x2 = ROOT.std.vector("float")(1)
x3 = ROOT.std.vector("float")(1)

label = ROOT.std.vector("int")(1)
weight = ROOT.std.vector("float")(1)

t.Branch("run", run)
t.Branch("lumi", lumi)
t.Branch("event", event)

t.Branch("x1", x1)
t.Branch("x2", x2)
t.Branch("x3", x3)

t.Branch("label", label)
t.Branch("weight", weight)

for i in range(50_000):

    run[0] = 1
    lumi[0] = i // 1000
    event[0] = i

    x1[0] = random.gauss(0, 1)
    x2[0] = random.gauss(0, 1)
    x3[0] = random.gauss(0, 1)

    score = x1[0] + 0.5 * x2[0] - x3[0]

    label[0] = 1 if score > 0.2 else 0
    weight[0] = random.uniform(0.5, 1.5)

    t.Fill()

t.Write()
f.Close()

print("Wrote synthetic.root")
