# -*- coding: utf-8 -*-
#Copy of mae.ipynb

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#this function takes in a image and masks 75% of the image
def mask_image(image, mask_ratio=0.75):

    # convert image to default size
    image = image.view(3, 32, 32)

    #mask a the right number of patches
    num_patches = image.shape[1] * image.shape[2]
    num_masked = int(mask_ratio * num_patches)
    mask = np.random.choice(num_patches, num_masked, replace=False)

    masked_image = image.clone()
    for idx in mask:
        h, w = divmod(idx, image.shape[2])
        masked_image[:, h, w] = 0

    #flatten image before returning it
    return masked_image.view(-1)

#this is the simple MAE model
class SimpleMAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleMAE, self).__init__()

        #intialize the encoder of MAE
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        #intialize the decoder of MAE
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):

        #send the masked image through the encoder
        encoded = self.encoder(x)

        #then reconstruct the image in the decoder
        decoded = self.decoder(encoded)
        return decoded

#this function take the iamges and prints them out
def plot_results(original, masked, reconstructed):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    #print original image
    original = original.view(3, 32, 32)
    axs[0].imshow(original.permute(1, 2, 0).numpy())
    axs[0].set_title("Original Image")

    #print masked image
    axs[1].imshow(masked.permute(1, 2, 0).numpy())
    axs[1].set_title("Masked Image")

    #print reconstucted image
    if reconstructed.dim() == 1:
        reconstructed = reconstructed.view(3, 32, 32)
    axs[2].imshow(reconstructed.detach().permute(1, 2, 0).numpy())
    plt.show()

#set parameters
input_dim = 32 * 32 * 3
hidden_dim = 128
learning_rate = 1e-3
num_epochs = 5
mask_ratio = 0.75

#get the CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = CIFAR10(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

#initialize model, loss, and optimizer
model = SimpleMAE(input_dim=input_dim, hidden_dim=hidden_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#main loop that does the masking and reconstruction
for epoch in range(num_epochs):
    for images, _ in tqdm(dataloader):

        #mask all the images
        images = images.view(-1, input_dim)
        masked_images = torch.stack([mask_image(img, mask_ratio) for img in images])

        #send the masked images into the model
        reconstructed = model(masked_images)
        loss = criterion(reconstructed, images)

        #update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #display a reconstructed image after each epoch along with the loss
    masked_image = masked_images[0]
    reconstructed_image = model(masked_image.unsqueeze(0))
    plot_results(images[0], masked_image.view(3, 32, 32), reconstructed_image.squeeze(0))
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")