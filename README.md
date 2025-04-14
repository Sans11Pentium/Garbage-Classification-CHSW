# 🗑️ Garbage Classifier Streamlit App

This is a simple web application built with Streamlit that allows users to classify an uploaded image of waste into one of six predefined categories. It utilizes a trained model in TFLite format to make real-time predictions directly in the browser.

## 🚀 Features

- Upload an image of waste material
- Automatically predicts the class of garbage
- Easy-to-use and interactive interface
- Fast and efficient inference using TensorFlow Lite

### 🛠️ MobileNet to TFLite Conversion for Edge Impulse (TinyML Deployment)

To deploy the trained MobileNet model on an edge device using **Edge Impulse**, it was converted to the TensorFlow Lite (TFLite) format. This enabled real-time inference on resource-constrained hardware as a TinyML application. 
Link to live inferencing on EdgeImpulse : https://smartphone.edgeimpulse.com/classifier.html

#### Steps Followed:
1. **Model Training**  
   This lightweight `MobileNetV2` model using TensorFlow/Keras trained on the garbage classification dataset is designed to be efficient and suitable for edge deployment.
2. **TFLite Conversion**  
   After training, model was converted to a `.tflite` format using the TFLite converter.
   
## 🧠 Supported Classes

The app classifies the uploaded image into one of the following categories:

1. **Glass**
2. **Paper**
3. **Cardboard**
4. **Plastic**
5. **Metal**
6. **Trash**

## 📦 Access the app 

Link is provided in the desription of this repository.
