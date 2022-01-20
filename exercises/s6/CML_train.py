import os
from pathlib import Path
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

path_root = Path(__file__).parent

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('exercises/s0_datasets', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('exercises/s0_datasets', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        # Now with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        
        # output so no dropout here
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

def train():
    model = Classifier()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 30
    steps = 0

    train_losses, val_losses, val_accuracy = [], [], []
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        else:
            ## TODO: Implement the validation pass and print out the validation accuracy
            with torch.no_grad():
                val_loss, accuracy = 0, 0
                for images, labels in testloader:
                    log_ps = model(images)
                    loss = criterion(log_ps, labels)
                    val_loss += loss.item()
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
            train_losses.append(running_loss/len(trainloader))
            val_losses.append(val_loss/len(testloader))
            val_accuracy.append(accuracy.item()/len(testloader))
            print(f"Epoch{e}: Training loss: {train_losses[e]}, Val loss: {val_losses[e]}, Val Accuracy: {val_accuracy[e]*100}%")


    torch.save(model.state_dict(), os.path.join(path_root,'CML-model.pth'))

    plt.figure()
    plt.plot(range(epochs), train_losses, label="train loss")
    plt.plot(range(epochs), val_losses, label="val loss")
    plt.plot(range(epochs), val_accuracy, label="val accuracy")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy/Loss")
    plt.title("Training and Validation")

if __name__ == "__main__":

    train()
