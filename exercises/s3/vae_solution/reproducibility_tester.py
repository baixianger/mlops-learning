import os
import torch
from pathlib import Path
import sys

if __name__ == "__main__":

    path_root = Path(__file__).parent
    print(path_root)
    model1 = torch.load(os.path.join(str(path_root), "model1.pth"))
    model2 = torch.load(os.path.join(str(path_root), "model2.pth"))
    
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2), \
            "encountered a difference in parameters, your script is not fully reproduceable"
