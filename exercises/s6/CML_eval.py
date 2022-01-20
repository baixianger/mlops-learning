import os
import sys
from pathlib import Path
path_root = Path(__file__).parent
sys.path.append(str(path_root))

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix

from CML_train import Classifier

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download and load the test data
testset = datasets.FashionMNIST('exercises/s0_datasets', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

model = Classifier()
state_dict = torch.load(os.path.join(path_root,'CML-model.pth'))
model.load_state_dict(state_dict)

model.eval()
preds, target = [], []
for batch in testloader:
    x, y = batch
    log_probs = model(x)
    y_hat = torch.exp(log_probs).argmax(dim=-1)
    preds.append(y_hat.detach())
    target.append(y.detach())
    print(y_hat.shape, y.shape)

target = torch.cat(target, dim=0)
preds = torch.cat(preds, dim=0)
report = classification_report(target, preds)
with open("classification_report.txt", 'w') as outfile:
    print(os.getcwd())
    outfile.write(report)
confmat = confusion_matrix(target, preds)
# disp = ConfusionMatrixDisplay(confmat)
plot_confusion_matrix(confmat)
plt.savefig("confusion_matrix.png")