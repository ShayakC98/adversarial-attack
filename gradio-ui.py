import gradio as gr
import requests
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO

def resize_image(image, target_size=(224, 224)):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = image.resize(target_size)
    return np.array(image)

def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode()

def classify(image, model_name, attack_method):
    # Resize image to 224x224
    image = resize_image(image, (224, 224))
    encoded_image = encode_image(image)
    
    # Call FastAPI for inference
    response = requests.post("http://127.0.0.1:8000/predict", json={
        "model_name": model_name,
        "attack_method": attack_method,
        "image": encoded_image
    })
    result = response.json()
    return result["prediction_original"], result["prediction_attacked"]

iface = gr.Interface(
    fn=classify,
    inputs=[
        gr.Image(type="numpy", label="Upload Image"),
        gr.Dropdown(choices=["resnet", "vgg19", "inceptionv3", "efficientnet"], label="Model"),
        gr.Dropdown(choices=["gaussian_blur", "linear_blur", "change_texture", "wave_distortion", "salt_and_pepper_noise", "fgsm_attack"], label="Adversarial Attack")
    ],
    outputs=[gr.Label(label="Original Prediction"), gr.Label(label="Attacked Prediction")],
    title="Adversarial Attack on Vision Models",
    description="Select a vision model and an adversarial attack to see the predictions."
)

iface.launch()

