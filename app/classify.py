import tensorflow as tf
import numpy as np
from PIL import Image

interpreter = tf.lite.Interpreter(model_path="model/garbage_classifier.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

categories = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']

def predict(image_path):
    img = Image.open(image_path).resize((100, 100)).convert("RGB")
    img = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return categories[np.argmax(output_data)]
