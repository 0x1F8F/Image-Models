import gradio as gr

# from gradio import component as grc
import numpy as np

# import plotly.express as ex
from PIL import Image
from loader import model, normalize
from torchvision.transforms import PILToTensor
import torch

transforms = PILToTensor()
model.eval()


def image_post_process(image: Image):
    image: torch.Tensor = transforms(image).to(torch.float).div_(255)
    print(image.unsqueeze(0).unsqueeze(0).shape)
    print(image)
    exit()
    with torch.no_grad():
        prediction: torch.Tensor = model(image)

    return dict(enumerate(prediction.view(-1).tolist()))


app = gr.Interface(
    fn=image_post_process,
    title="HandDigit prediction",
    inputs=gr.Sketchpad(type="pil"),
    outputs=gr.Label(),
    live=True
)

app.launch(server_name="0.0.0.0", server_port=8080)
