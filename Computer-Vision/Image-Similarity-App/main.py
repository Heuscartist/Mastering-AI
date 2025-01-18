import streamlit as st
import os
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
import pickle

# This script is used to create the streamlit application that is used for the image similarity task
# The application allows the user to upload an image and then finds the most similar image from the database
# The similarity is computed using the cosine similarity between the embeddings of the images
# The application uses pretrained VGG16 Model to generate embeddings for the images
# The top is not included to get the embeddings from the last hidden layer
# For faster retrieval of embeddings, the embeddings are computed and saved into a pkl file
# Without storage of the embeddings, the time taken for computing the embeddings for each image was very high
# The image similarity is computed using the cosine similarity between the embeddings of the images
# The most similar image is then displayed to the user

# Loading the VGG16 Model using the imagenet weights and excluding the top layer
vgg16 = VGG16(weights="imagenet", include_top=False)

# Folder containing the image database (q3_database)
image_folder = 'q3_database'
image_database = os.listdir(image_folder)

# The image_embeddigns.pkl file contains the embeddings of the images in the database
# the pkl file is computed and saved using the compute_embeddings.py script
embeddings_file = 'image_embeddings.pkl'

# Utility Functions

# Function to load and resize image according to the VGG16 model input dimensions
def load_image(image_path):
    """Loads and resizes an image to 244x244."""
    input_image = Image.open(image_path)
    resized_image = input_image.resize((244, 244))
    return resized_image

# Function to use the VGG16 models to generate embeddings for input image
def get_image_embeddings(object_image):
    """Generates embeddings for the given image using VGG16."""
    image_array = np.expand_dims(keras_image.img_to_array(object_image), axis=0)
    image_array = preprocess_input(image_array)
    image_embedding = vgg16.predict(image_array)
    return image_embedding.flatten()

# Function foe loading precomputed embeddings or computing and saving them in case file not found
def load_or_compute_embeddings():
    """Loads precomputed embeddings or computes and saves them."""
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        embeddings = {}
        for img_name in image_database:
            image_path = os.path.join(image_folder, img_name)
            image = load_image(image_path)
            embeddings[img_name] = get_image_embeddings(image)
        
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)
    
    return embeddings

# Streamlit Application
def main():
    st.title("Image Similarity Finder")

    st.sidebar.header("Instructions")
    st.sidebar.markdown("1. Upload an image.")
    st.sidebar.markdown("2. Click the 'Find Similar Images' button to find the most similar image from the database.")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Find Similar Images"):
            uploaded_image_path = "temp_uploaded_image.jpg"
            with open(uploaded_image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

            uploaded_image_obj = load_image(uploaded_image_path)
            uploaded_image_embedding = get_image_embeddings(uploaded_image_obj)

            embeddings = load_or_compute_embeddings()

            # Find the most similar image
            max_similarity_score = 0
            most_similar_image_path = None
            # Looping over the embeddings to find the most similar image using cosine similarity
            for image_name, embedding in embeddings.items():
                similarity_score = cosine_similarity([uploaded_image_embedding], [embedding]).flatten()[0]
                if similarity_score > max_similarity_score:
                    max_similarity_score = similarity_score
                    most_similar_image_path = os.path.join(image_folder, image_name)

            # Display the most similar image
            if most_similar_image_path:
                st.success(f"Most similar image found with a similarity score of {max_similarity_score:.2f}")
                st.image(most_similar_image_path, caption="Most Similar Image", use_column_width=True)

            # Clean up temporary file
            os.remove(uploaded_image_path)

if __name__ == "__main__":
    main()
