#!/usr/bin/env python
# coding: utf-8

# In[73]:


import sklearn
from sklearn.datasets import make_circles


# In[74]:


# Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)


# In[75]:


len(X), len(y)


# In[76]:


print(f'First 5 samples of X:\n {X[:5]}')
print(f'First 5 samples of y:\n {y[:5]}')


# In[77]:


# Make DataFrame of circle data
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})
circles.head(10)


# In[78]:


import matplotlib.pyplot as plt
plt.scatter(x=X[:, 0], y=X[:, 1], c = y, cmap=plt.cm.RdYlBu)


# In[79]:


# check input and output shapes
X.shape, y.shape


# In[80]:


# View the first example of features and labels
X_sample = X[0]
y_sample = y[0]
print(f'Values for one sample of X: {X_sample} and the same for y: {y_sample}')
print(f'Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}')


# In[81]:


type(X), type(y)


# In[82]:


# Turn data into tensors and create train and test splits
import torch
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)


# In[83]:


X[:5], y[:5]


# In[84]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)


# In[85]:


len(X_train), len(X_test), len(y_train), len(y_test)


# In[86]:


#building a model
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
device


# In[87]:


class CircleModelV0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Create nn.Linear layers
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))

model_0 = CircleModelV0().to(device=device)
model_0


# In[88]:


device


# In[89]:


next(model_0.parameters()).device


# In[90]:


#replicating the model using nn.Sequential()
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device=device)

model_0


# In[94]:


#Make predictions

untrained_preds = model_0(X_test.to(device=device))
print(f'Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}')
print(f'Length of test samples: {len(X_test)}, Shape: {X_test.shape}')
print(f'\nFirst predictions:\n{untrained_preds[:10]}')
print(f'\nFirst 10 labels:\n{y_test[:10]}')


# In[ ]:




