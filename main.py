# ---------------------------------------------
# MNIST handwritten digit recognition with PyTorch
# ---------------------------------------------

# Author: Han Wang
# Date: 2025-05-01
# Description: This script contains an implementation of a CNN for image classification using PyTorch. The model is trained on the MNIST dataset.

# Import necessary libraries

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import torch
from torch import nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.nn.functional import one_hot
from torchsummary import summary
from torchmetrics.classification import Accuracy, Precision, Recall

#%matplotlib inline
np.random.seed(1)

# Load the MNIST dataset

## Download and load the data:
train_data = MNIST(root="./mnist-data", train=True, download=True)
test_data = MNIST(root='./mnist-data', train=False, download=True)

print("Training set size:", len(train_data))
print("Test set size:", len(test_data))

## Manually check some images:

fig, axes = plt.subplots(1, 3, figsize=(10, 3))  

for ax, i in zip(axes, np.random.randint(0, 60000, 3)):
    image, label = train_data[i]
    ax.imshow(image, cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')

plt.tight_layout()
plt.show()

## Transform the data to tensors:

### Define a transform to convert the images to tensors:
transform = transforms.Compose([
    transforms.ToTensor()
])

### Apply the transform to the training and test datasets:
train_data = MNIST(root='./mnist-data', train=True, transform=ToTensor(), download=False)
test_data = MNIST(root='./mnist-data', train=False, transform=ToTensor(), download=False)


## Prepare the validation set:
### Split the testing data further into validation and test sets:

from torch.utils.data import random_split

cv_size = int(0.5 * len(test_data))  # 50% for CV
test_size = len(test_data) - cv_size  # Remaining 50% for Test

cv_data, test_data = random_split(test_data, [cv_size, test_size])

### Transform the data shape and use one-hot encoding for labels:

def transform_shape(data_set):
    
    data_examples = torch.zeros(size=(len(data_set), 28, 28))
    targets = torch.zeros(size=(len(data_set), 1))
    
    for i,instance in enumerate(data_set):
        data_examples[i] = instance[0].reshape(28,28) 
        targets[i] = instance[1]
        
    targets = one_hot(targets.long(), 10).squeeze()
        
    return data_examples, targets

new_train_x, new_train_y = transform_shape(train_data)
new_test_x, new_test_y = transform_shape(test_data)
new_cv_x, new_cv_y = transform_shape(cv_data)


# Modelling:

### Make device agnostic code
def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")           # use "cuda:0" for a specific GPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = default_device()


## CNN Model definition:

def cnn_model():
    """
    Implements the forward propagation for the multiclass classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MaxPool2D -> FLATTEN -> LazyLinear -> Softmax
    
    Arguments:
    None

    Returns:
    model -- PyTorch Sequential container
    """
    model = nn.Sequential(
              nn.ZeroPad2d(2),
              nn.Conv2d(1, 16, 5, 1),
              nn.BatchNorm2d(16),
              nn.ReLU(),
              nn.MaxPool2d(2),
              nn.Flatten(),
              nn.LazyLinear(out_features=10),
              nn.Softmax(dim=1)
            )
    return model

## Model dry run:

### Pass some data through the model:

#### Instantiate the model
model = cnn_model()

#### Ensure the model is on the correct device
model = model.to(device)

summary(model, (1, 28, 28), device = device.type)

#### Make predictions
untrained_preds = model(new_cv_x[:5].unsqueeze(dim = 1).to(device))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(new_cv_y[:5])}, Shape: {new_cv_y[:5].shape}")
print(f"\nFirst 5 predictions:\n{torch.round(untrained_preds[:5])}")
print(f"\nFirst 5 test labels:\n{new_cv_y[:5]}")


## Model configuration:
### Define the loss function and optimizer:

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

### Define performance metrics:

num_classes = 10  # 10-classes

train_accuracy = Accuracy(num_classes=num_classes, task="multiclass").to(device)
train_precision = Precision(num_classes=num_classes, task="multiclass").to(device)
train_recall = Recall(num_classes=num_classes, task="multiclass").to(device)

cv_accuracy = Accuracy(num_classes=num_classes, task="multiclass").to(device)
cv_precision = Precision(num_classes=num_classes, task="multiclass").to(device)
cv_recall = Recall(num_classes=num_classes, task="multiclass").to(device)

### Set batch size amd send training and validation data to device:

from torch.utils.data import TensorDataset, DataLoader

new_train_x, new_train_y = new_train_x.to(device), (new_train_y.type(torch.float)).to(device)
new_cv_x, new_cv_y = new_cv_x.to(device), (new_cv_y.type(torch.float)).to(device)

dataset = TensorDataset(new_train_x, new_train_y)

# Define batch size
batch_size = 32

# Create DataLoader
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


## Train the model:

def train_model(my_model, epoch_nums, optimizer, loss, data_loader, new_cv_x, new_cv_y):


    # Set the number of epochs
    epochs = epoch_nums

    # Build training and evaluation loop
    for epoch in range(epochs):
        ### Training
        my_model.train()

        epoch_train_loss = 0
        epoch_cv_loss = 0

        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # 1. Forward pass (model outputs raw logits)
            y_logits = my_model(inputs.unsqueeze(dim = 1)).squeeze() # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device 
            y_pred = torch.round(y_logits) # turn logits -> pred probs -> pred labls

            # 2. Calculate loss/accuracy
             
            train_loss = loss(y_logits, # Using nn.BCEWithLogitsLoss works with raw logits
                       targets) 
            epoch_train_loss = train_loss

            # 3. Update metrics
            train_accuracy.update(y_pred, targets)
            train_precision.update(y_pred, targets)
            train_recall.update(y_pred, targets)

            # 4. Optimizer zero grad
            optimizer.zero_grad()

            # 5. Loss backwards
            train_loss.backward()

            # 6. Optimizer step
            optimizer.step()

            ### Testing
            my_model.eval()
            with torch.inference_mode():
                # 1. Forward pass
                cv_logits = my_model(new_cv_x.unsqueeze(dim = 1)).squeeze() 
                cv_pred = torch.round(cv_logits)
                # 2. Caculate loss/accuracy
                cv_loss = loss(cv_logits,
                                    new_cv_y)
                epoch_cv_loss = cv_loss
                cv_accuracy.update(cv_pred, new_cv_y)
                cv_precision.update(cv_pred, new_cv_y)
                cv_recall.update(cv_pred, new_cv_y)

        # Compute metrics at the end of the epoch
        epoch_train_accuracy = train_accuracy.compute()
        epoch_train_precision = train_precision.compute()
        epoch_train_recall = train_recall.compute()

        epoch_cv_accuracy = cv_accuracy.compute()
        epoch_cv_precision = cv_precision.compute()
        epoch_cv_recall = cv_recall.compute()


        # Print out what's happening every epoch
        
        print(f"Epoch: {epoch} | Train Loss: {epoch_train_loss:.4f} | CV loss: {epoch_cv_loss:.4f} \nTrain Accuracy: {epoch_train_accuracy.item():.4f}, Train Precision: {epoch_train_precision.item():.4f}, Train Recall: {epoch_train_recall.item():.4f}\nCV Accuracy: {epoch_cv_accuracy:.4f}, CV Precision: {epoch_cv_precision:.4f}, CV Recall: {epoch_cv_recall:.4f}\n\n")

        # Reset metrics for the next epoch
        train_accuracy.reset()
        train_precision.reset()
        train_recall.reset()

        cv_accuracy.reset()
        cv_precision.reset()
        cv_recall.reset()

train_model(model, 20, optimizer, loss, data_loader, new_cv_x, new_cv_y)


# Test the model:

## Test metrics:
test_accuracy = Accuracy(num_classes=num_classes, task="multiclass").to(device)
test_precision = Precision(num_classes=num_classes, task="multiclass").to(device)
test_recall   = Recall(num_classes=num_classes,   task="multiclass").to(device)

## Test the model with inference mode:
model.eval()
with torch.inference_mode():
    # forward pass on the whole test tensor
    test_logits = model(new_test_x.to(device).unsqueeze(1)).squeeze()
    test_preds  = torch.round(test_logits)

    # loss & metric updates
    targets = new_test_y.to(device).type(torch.float)
    test_loss = loss(test_logits, targets)
    test_accuracy.update(test_preds, targets)
    test_precision.update(test_preds, targets)
    test_recall.update(test_preds, targets)

## Print test metrics:
print(f"loss      : {test_loss.item():.4f}")
print(f"accuracy  : {test_accuracy.compute():.4f}")
print(f"precision : {test_precision.compute():.4f}")
print(f"recall    : {test_recall.compute():.4f}")


# Save model weights to check point:
ckpt_path_gpu = "./checkpoint/250502_mnist_cnn_pt1_gpu.pt"
ckpt_path_mps = "./checkpoint/250501_mnist_cnn_pt1_mps.pt"
ckpt_path_cpu = "./checkpoint/250501_mnist_cnn_pt1_cpu.pt"
torch.save(model.to(device).state_dict(), ckpt_path_gpu) # Save the model weights to its onriginal device (MPS in my case).
torch.save(model.to("cpu").state_dict(), ckpt_path_cpu) # Save the model weights to CPU so it's device agnostic.
print(f"\n Weights saved to {ckpt_path_gpu} and {ckpt_path_cpu}.\n")