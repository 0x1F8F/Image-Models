#!/bin/env python

import gradio as gr
import numpy as np
from PIL import Image
from loader import model, normalize
from torchvision.transforms import PILToTensor
import torch

transforms = PILToTensor()
model.eval()


def image_post_process(image: Image):
    if isinstance(image, Image.Image):
        image: torch.Tensor = transforms(image).to(torch.float).div_(255)
        with torch.no_grad():
            prediction: torch.Tensor = model(image)

        return dict(enumerate(prediction.view(-1).tolist()))
    return {"display": 1.0}


app = gr.Interface(
    fn=image_post_process,
    title="HandDigit prediction",
    inputs=gr.Sketchpad(type="pil"),
    outputs=gr.Label(),
    live=True,
)

app.launch(server_name="0.0.0.0", server_port=8080)
