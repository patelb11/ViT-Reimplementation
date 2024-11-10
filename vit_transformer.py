# -*- coding: utf-8 -*-
#ViT_transformer.ipynb

!pip install torch torchvision

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10, CIFAR100, StanfordCars
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTFeatureExtractor

#GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class PatchEncoding(nn.Module):
  def __init__(self, img_size, patch_size, in_channels, embedded_dimensions):
    super(PatchEncoding, self).__init__()

    #set variables
    self.img_size = img_size
    self.patch_size = patch_size
    self.num_patches = (img_size // patch_size) ** 2
    self.embed_dim = embedded_dimensions

    #do the linear projection of flattened images
    self.patch_creation = nn.Conv2d(in_channels, embedded_dimensions, kernel_size=patch_size, stride=patch_size)
    self.cls_token = nn.Parameter(torch.zeros(1, 1, embedded_dimensions))
    self.flatten = nn.Flatten(2)

  def forward(self, x):

    #get the variables from the image
    batch_size, channel, height, width = x.shape

    #reshape the image to be the correct input to the transformer
    x = self.patch_creation(x)
    x = self.flatten(x)
    x = x.transpose(1, 2)

    #create the cls token to the image patches
    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    return x

class TransformerEncoder(nn.Module):
  def __init__(self, embedded_dimensions, num_heads, mlp_dimensions, dropout_rate):
    super(TransformerEncoder, self).__init__()

    #define all the layers in the encoder
    self.norm1 = nn.LayerNorm(embedded_dimensions)
    self.attention_layer = nn.MultiheadAttention(embedded_dimensions, num_heads, dropout=dropout_rate)
    self.norm2 = nn.LayerNorm(embedded_dimensions)
    self.mlp = nn.Sequential(
        nn.LayerNorm(embedded_dimensions),
        nn.Linear(embedded_dimensions, mlp_dimensions),
        nn.GELU(),
        nn.Linear(mlp_dimensions, embedded_dimensions),
        nn.Dropout(dropout_rate))

  def forward(self, x):

    x = self.norm1(x)
    x_attention, _ = self.attention_layer(x, x, x)
    x = x + x_attention
    x = x + self.mlp(self.norm2(x))

    return x

class ViT(nn.Module):
  def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_dim, dropout_rate=0.1):
    super(ViT, self).__init__()

    #get the patches
    self.patch_embedding = PatchEncoding(img_size, patch_size, in_channels, embed_dim)

    #get positional embeddings for the image patches
    self.pos_embxedding = nn.Parameter(torch.randn(1, 1 + self.patch_embedding.num_patches, embed_dim))

    #do the transformer encoder
    self.transformer = nn.ModuleList(
        [TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout_rate) for _ in range(depth)]
    )

    self.layer_norm = nn.LayerNorm(embed_dim)

    #create the mlp head
    self.mlp_head = nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, num_classes)
    )

  def forward(self, x):
    x = self.patch_embedding(x)
    x = x + self.pos_embedding
    for transformer in self.transformer:
      x = transformer(x)
    x = self.layer_norm(x[:, 0])
    x = self.mlp_head(x)
    return x

#create testing and training functions
def train(model, dataloader, criterion, optimizer):
  model.train()

  running_loss = 0.0

  #loop though data
  for inputs, labels in tqdm(dataloader):

      # Move data to the device
      inputs, labels = inputs.to(device), labels.to(device)

      #apply optimizer
      optimizer.zero_grad()

      #find outputs and calculate loss
      outputs = model(inputs)
      loss = criterion(outputs.logits, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

  return running_loss / len(dataloader)

def test(model, dataloader, criterion):
  model.eval()

  #intliaize variables
  running_loss = 0.0
  correct = 0
  total = 0

  with torch.no_grad():
      #loop through all data
      for inputs, labels in tqdm(dataloader):

          #move data to the device
          inputs, labels = inputs.to(device), labels.to(device)

          #get outputs and get accuracy
          outputs = model(inputs)
          loss = criterion(outputs.logits, labels)
          running_loss += loss.item()
          _, predicted = outputs.logits.max(1)
          total += labels.size(0)
          correct += predicted.eq(labels).sum().item()

  accuracy = 100.0 * correct / total
  return running_loss / len(dataloader), accuracy

def get_data_loaders(dataset_name, batch_size):

  #initialize a series of transforms to applt to the images
  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
  ])

  #load the dataset you want to train and test on
  if dataset_name == "CIFAR10":
      #CIFAR10 - 10 classes
      train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
      test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
  elif dataset_name == "CIFAR100":
      #CIFAR100 - 100 classes
      train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
      test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
  elif dataset_name == "StanfordCars":
      # StanfordCars dataset, split into training and testing sets
      train_dataset = datasets.StanfordCars(root='./data', split='train', download=True, transform=transform)
      test_dataset = datasets.StanfordCars(root='./data', split='test', download=True, transform=transform)

  #create dataloader
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  return train_loader, test_loader

#main function

#set the parameters for the ViT
params = {
    'img_size': 224,
    'patch_size': 16,
    'in_channels': 3,
    'num_classes': 100,
    'embed_dim': 768,
    'depth': 12,
    'num_heads': 12,
    'mlp_dim': 3072,
    'dropout_rate': 0.1
}

#initalize the ViT transformer
# model = ViT(**params).to(device)
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=100, ignore_mismatched_sizes=True).to(device)

#set the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#set training and testing variables
batch_size = 16
dataset_name = "CIFAR100"
epochs = 20

#get the training and testing data
train_loader, test_loader = get_data_loaders(dataset_name, batch_size)

#training and testing for loop
for epoch in range(epochs):
  print(f"Epoch {epoch+1}/{epochs}")
  train_loss = train(model, train_loader, criterion, optimizer)
  test_loss, test_accuracy = test(model, test_loader, criterion)
  print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

