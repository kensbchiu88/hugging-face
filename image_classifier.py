####
# todo 
####

import gradio as gradio
from transformers import pipeline
from PIL import Image
import requests

# load pipe
#image_classifier = pipeline("zero-shot-image-classification", model="google/siglip-so400m-patch14-384")
image_classifier = pipeline(task="zero-shot-image-classification", model="openai/clip-vit-base-patch32")

# label
candidate_labels=["green", "red", "yellow", "blue", "none"]

def predict(input_image):
    outputs = image_classifier(input_image, candidate_labels=["green", "red", "yellow", "blue", "none"])
    return outputs

iface = gradio.Interface(fn=predict, inputs=gradio.Image(label="Upload an image", type="pil"), outputs="text")

if __name__ == "__main__":
    iface.launch(share=False)


