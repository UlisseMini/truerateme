# %%

import json
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import requests
from tqdm import tqdm
from PIL import Image
import torch
import os
import plotly.express as px

# %%

with open("data.json") as f:
    data = json.load(f) # url: [ratings]

# %%
# Load clip

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Get all the vectors and embed them using clip
# %%

def remove_nan(dataset):
    return [(emb, rating) for emb, rating in dataset if not torch.isnan(emb).any() and not np.torch(rating).any()]


def load_dataset():
    dataset = []

    for url, ratings in tqdm(data.items()):
        # TODO: 404
        try:
            image = Image.open(requests.get(url, stream=True).raw)
        except Exception as e:
            continue

        # image = torch.tensor(np.array(image), device='mps') # TODO
        inputs = processor(images=image, return_tensors="pt", padding=True)
        image_features = model.get_image_features(**inputs)
        rating = torch.tensor(np.median(ratings), dtype=torch.float32)
        dataset.append((image_features, rating))

    dataset = remove_nan(dataset)
    return dataset


if not os.path.exists('dataset.pt'):
    dataset = load_dataset()
    torch.save(dataset, "dataset.pt")
else:
    dataset = torch.load('dataset.pt')


# Load images from ./faces
# TODO


# %%
# Create train and test sets

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# %%
# Create a dataloader for minibatching

batch_size = 32
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# %%

def mean_loss(model, dataloader):
    total_loss = 0.
    with torch.no_grad():
        for image_features, rating in dataloader:
            prediction = model(image_features)
            loss = ((prediction.squeeze() - rating)**2).mean()
            total_loss += loss.item()

    return total_loss / len(dataloader)


# %%
# Initialize model

trm_model = torch.nn.Linear(512, 1)
optim = torch.optim.AdamW(
    trm_model.parameters(),
    lr=0.01,
    weight_decay=0.1
)

# %%
# Train model

epochs = 1000
pbar = tqdm(total=epochs)
for epoch in range(epochs):
    for image_features, rating in train_dataloader:
        optim.zero_grad()
        prediction = trm_model(image_features)
        # loss = torch.nn.functional.mse_loss(prediction, rating.unsqueeze(1))
        loss = ((prediction.squeeze() - rating)**2).mean()
        loss.backward()
        optim.step()

    # Evaluate
    test_loss = mean_loss(trm_model, test_dataset)
    train_loss = mean_loss(trm_model, train_dataset)

    pbar.set_description(f"test_loss: {test_loss:.4f} train_loss {train_loss:.4f}")
    pbar.update(1)

pbar.close()

# %%
# Save model

torch.save(trm_model.state_dict(), "model.pt")

# %%

faces = [Image.open(requests.get(url, stream=True).raw) for url in tqdm(data.keys())]

# %%
# Visualize images with their predictions and true ratings

from IPython.display import display

def visualize(start: int, end: int):
    for face, (image_features, rating) in list(zip(faces, dataset))[start:end]:
        # inputs = processor(images=face, return_tensors="pt", padding=True)
        # image_features = model.get_image_features(**inputs)
        prediction = trm_model(image_features)
        display(face)
        print(f"prediction: {prediction.item():.2f} true: {rating.item()}")


visualize(0, 10)

# %%


sorted(list(data.values())[1])

# %%

url = list(data.keys())[0]
print(url)
Image.open(requests.get(url, stream=True).raw)


