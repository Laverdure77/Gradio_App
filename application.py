import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from gradio import components
from PIL import Image

# List of rock classes
class_names = ['Conglomerate', 'Gneiss', 'Granite', 'Limestone', 'Pegmatite', 'Sandstone', 'Shale', 'Shoshonite porphyry']

# Load the pre-trained model
model = load_model(r"model.h5", compile=False)

# Compile the model
learning_rate = 0.0001
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

def preprocess_image(img, crop_option):
    # Get the original image size
    width, height = img.size

    if crop_option == "Square crop":  # Square crop on smallest dimension
        # Calculate the size of the centered square crop
        size = min(width, height)
        # Calculate the coordinates for the crop
        left = (width - size) // 2
        top = (height - size) // 2
        right = (width + size) // 2
        bottom = (height + size) // 2
        # Crop the image
        img = img.crop((left, top, right, bottom))
    
    elif crop_option == "600x600 center crop":  # 600x600 center crop
        # Calculate the coordinates for the crop
        left = (width - 600) // 2
        top = (height - 600) // 2
        right = left + 600
        bottom = top + 600
        # Crop the image
        img = img.crop((left, top, right, bottom))
    
    elif crop_option == "300x300 center crop":  # 300x300 center crop
        # Calculate the coordinates for the crop
        left = (width - 300) // 2
        top = (height - 300) // 2
        right = left + 300
        bottom = top + 300   
        # Crop the image
        img = img.crop((left, top, right, bottom))

    # Resize the cropped image to 299x299
    img = img.resize((299, 299))
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def predict(image, crop_option):
    preprocessed_image = preprocess_image(image, crop_option)
    predictions = model.predict(preprocessed_image)
    probabilities = predictions[0]

    sorted_indices = np.argsort(probabilities)[::-1]  # Sort indices in descending order
    top_indices = sorted_indices[:3]  # Select the top 3 indices
    top_class_names = [class_names[i] for i in top_indices]
    top_probabilities = [probabilities[i] for i in top_indices]

    return (
        {class_name: float(probability) for class_name, probability in zip(top_class_names, top_probabilities)},
        preprocessed_image[0],  # Return the preprocessed image as the second output
    )

article = f"""# Welcome to the MinersAI Rock Classification Deep Learning App!

## Introduction

Are you fascinated by the diverse world of rocks ? Our app brings cutting-edge AI technology to the realm of geology, allowing you to explore and classify various types of rocks with ease.

## How it Works

Using the power of advanced image recognition algorithms, our app can analyze images of rocks and identify their types based on distinct features such as texture, color, patterns, and more. Whether you're an amateur rock enthusiast, a student, or a geology professional, this app is designed to assist you in classifying rocks efficiently.

## Key Features

- **Image Upload:** Simply upload an image of the rock you want to classify.
- **Crop Options:** Choose how you want to crop the image before resizing.
- **Instant Analysis:** Receive real-time results and insights about the rock's classification.
- **User-Friendly Interface:** Our intuitive interface makes rock classification easy and accessible to all users.
- **Educational Tool:** Perfect for learning about different rock types and their distinguishing features.

## Getting Started

To start classifying rocks using our app, follow these steps:

1. Upload an image of the rock.
2. Select a crop option using the radio box.
3. Click on the "Submit" button to start the classification.
4. Explore the classification result.

## Rock Types

Here are the rock types that our app can classify:

{", ".join(class_names)}

## Embrace the World of Geology

Join us in unraveling the mysteries of rocks through the lens of AI. Whether you're identifying rocks you've collected or simply curious about the geological makeup of the world around you, our Rock Classification Computer Vision App is here to assist you.

Get ready to discover the incredible stories behind every rock! 🌍🔬

[MinersAI Homepage](https://www.minersai.com/)
"""

application = gr.Interface(
        fn=predict,
        inputs=[
            components.Image(type="pil", label="Upload an image", width=1000, height=600),
            components.Radio(label="Crop Options", choices=[
                "No crop",
                "Square crop",
                "600x600 center crop",
                "300x300 center crop",
            ], value="No crop"),
        ],
        outputs=[
            components.Label(label="Predictions", width=400, height=100),
            components.Image(type="numpy", label="Preprocessed Image", width=320, height=320)
        ],
        article=article,
        theme=gr.themes.Base(),
        allow_flagging="never"
    )


if __name__ == "__main__":
    application.launch(share=False, server_name="0.0.0.0", server_port=7860, auth=[("minersai", "minersai"),("admin", "minersai")], auth_message="Welcome to MinersAI Rock Classification Tool, please login below :")