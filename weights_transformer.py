import torch

PATHTOLOAD = ""
PATHTOSAVE = ""

model = torch.load(PATHTOLOAD, map_location="cuda:0")["model"]
torch.save({"model": model}, PATHTOSAVE)