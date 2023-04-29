# %%

import torch
import numpy as np
import gradio as gr
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load models
# %%

trm_model = torch.nn.Linear(512, 1)
trm_model.load_state_dict(torch.load("model.pt"))
trm_model.eval()

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# %%

def rateme(input_img):
    # TODO: figure out what bullshit clip is doing
    image = Image.fromarray(input_img)
    inputs = processor(images=image, return_tensors="pt", padding=True)
    image_features = model.get_image_features(**inputs)
    rating = trm_model(image_features).item()
    return f'you are a {rating:.1f}'


demo = gr.Interface(rateme, gr.Image(source="webcam", stream=True), "text")
demo.launch()
