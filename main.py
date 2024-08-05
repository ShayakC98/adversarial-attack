import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import base64

app = FastAPI()

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)
    
def apply_linear_blur(image, kernel_size=(5, 5)):
    return cv2.blur(image, kernel_size)

def change_texture(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    high_pass = cv2.Laplacian(gray, cv2.CV_64F)
    high_pass = np.uint8(np.absolute(high_pass))
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    texture_image = cv2.addWeighted(high_pass, 0.5, noise, 0.5, 0)
    return cv2.merge([texture_image, texture_image, texture_image])

def add_wave_distortion(image):
    rows, cols, ch = image.shape
    for i in range(rows):
        image[i, :] = np.roll(image[i, :], int(5.0 * np.sin(2 * np.pi * i / 150)))
    return image

def salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy = np.copy(image)
    num_salt = np.ceil(salt_prob * image.size)
    num_pepper = np.ceil(pepper_prob * image.size)

    # Add salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 1

    # Add pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 0

    return noisy

def fgsm_attack(image, model, epsilon=0.01):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.categorical_crossentropy(y_true=tf.one_hot(tf.argmax(prediction, axis=1), depth=1000), y_pred=prediction)
    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    perturbed_image = image + epsilon * signed_grad
    return perturbed_image.numpy()

class InferenceRequest(BaseModel):
    model_name: str
    attack_method: str
    image: str

models = {
    "resnet": tf.keras.applications.ResNet50(weights='imagenet'),
    "vgg19": tf.keras.applications.VGG19(weights='imagenet'),
    "inceptionv3": tf.keras.applications.InceptionV3(weights='imagenet'),
    "efficientnet": tf.keras.applications.EfficientNetB0(weights='imagenet')
}

def preprocess_image(image, model_name):
    if model_name in ["resnet", "vgg19"]:
        image = tf.keras.applications.resnet50.preprocess_input(image)
    elif model_name == "inceptionv3":
        image = tf.keras.applications.inception_v3.preprocess_input(image)
    elif model_name == "efficientnet":
        image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image

def decode_image(image_str):
    image_data = base64.b64decode(image_str)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def predict(image, model_name):
    model = models[model_name]
    image = np.expand_dims(image, axis=0)
    image = preprocess_image(image, model_name)
    preds = model.predict(image)
    return tf.keras.applications.imagenet_utils.decode_predictions(preds, top=1)[0]

def apply_attack(image, attack_method, model=None):
    if attack_method == "gaussian_blur":
        return apply_gaussian_blur(image)
    elif attack_method == "linear_blur":
        return apply_linear_blur(image)
    elif attack_method == "change_texture":
        return change_texture(image)
    elif attack_method == "wave_distortion":
        return add_wave_distortion(image)
    elif attack_method == "salt_and_pepper_noise":
        return salt_and_pepper_noise(image)
    elif attack_method == "fgsm_attack":
        return fgsm_attack(image, model)
    else:
        return image

@app.post("/predict")
async def predict_image(request: InferenceRequest):
    model_name = request.model_name
    attack_method = request.attack_method
    image_str = request.image
    
    # Decode base64 string to image
    image = decode_image(image_str)
    
    # Apply adversarial attack
    attacked_image = apply_attack(image, attack_method, models[model_name])
    
    prediction_original = predict(image, model_name)
    prediction_attacked = predict(attacked_image, model_name)

    print(prediction_original)
    print(prediction_attacked)
    
    return {
        "prediction_original": prediction_original[0][1],
        "prediction_attacked": prediction_attacked[0][1]
    }
