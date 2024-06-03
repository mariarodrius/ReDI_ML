#%% packages
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import seaborn as sns
#%% Hyperparameters
LR = 0.001
BS = 8
EPOCHS = 1000

# %% data prep
iris = load_iris()
X = iris.data.astype(np.float32)
y_batch = iris.target.astype(np.int64)

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(
     X, y_batch, test_size=0.2, random_state=42)

# %% Dataset and Dataloader
class IrisData(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.len = self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.len
    
iris_data = IrisData(X=X_train, y=y_train)
train_loader = DataLoader(dataset=iris_data, batch_size=BS)

# %% define network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(in_features=4, out_features = 6)
        self.lin2 = nn.Linear(in_features=6, out_features=3)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.softmax(x)
        return x  

# %% Model instance
model = Net()


# %% Loss function
loss_fn = nn.CrossEntropyLoss()

##% Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# %% Train Loop
losses = []
for epoch in range(EPOCHS):
    loss_batch = []
    print(f"Epoch: {epoch}")
    for x_batch, y_batch in train_loader:
        
        # initialize gradients
        optimizer.zero_grad()
        
        # forward pass
        y_pred = model(x_batch)
        
        # calculate losses
        
        loss = loss_fn(y_pred, y_batch)
        
        # update weights
        loss.backward()
        optimizer.step()
        
        # store loss
        loss_batch.append(loss.item())
    # add average loss for each epoch
    losses.append(sum(loss_batch)/len(loss_batch))
    
        

# %% study the losses
sns.lineplot(x=range(len(losses)), y=losses)

# %% test the model
X_test_torch = torch.from_numpy(X_test)
with torch.no_grad():
    y_test_pred_softmax = model(X_test_torch)
    y_test_hat = torch.argmax(y_test_pred_softmax, dim=1)
    print(f"Accuracy: {sum(y_test_hat.numpy() == y_test)/len(y_test)}")
    print(f"Predictions: {y_test_hat.numpy()}")
    print(f"True: {y_test}")

# %%
