import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
import pickle



# The purpose of this script is to first compute and store the embeddings of the images database
# This results in a faster retrieval of embeddings when the main.py script is run

# Same as the main.py file we will be loading the VGG16 model that has been pretrained on the imagenet dataset
# The top is set to false as we dont need to classify the images and instead need the vector embeddings
# from the last hidden layer
vgg16 = VGG16(weights="imagenet", include_top=False)

# Set the folder where the images are stored (image databse)
image_folder = 'q3_database'
image_database = os.listdir(image_folder)


# Utiliy functions

# load the image and resize it to 244x244 which is the input dimension set for the VGG16 model
def load_image(image_path):
    input_image = Image.open(image_path)
    resized_image = input_image.resize((244, 244))
    return resized_image

# Get the embeddings of the image using the VGG16 model
def get_image_embeddings(object_image):
    image_array = np.expand_dims(keras_image.img_to_array(object_image), axis=0)
    image_array = preprocess_input(image_array)
    image_embedding = vgg16.predict(image_array)
    return image_embedding.flatten()

# Compute and save the embeddings of the images into the pkl file
def compute_and_save_embeddings():
    embeddings = {}
    for img_name in image_database:
        image_path = os.path.join(image_folder, img_name)
        image = load_image(image_path)
        embeddings[img_name] = get_image_embeddings(image)
    
    with open('image_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    
    print("Embeddings computed and saved successfully!")

if __name__ == "__main__":
    compute_and_save_embeddings()
