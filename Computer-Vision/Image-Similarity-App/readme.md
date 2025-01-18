# Creating  Streamlit Application for Image Similarity
This script creates a Streamlit application for an image similarity task, enabling users to upload an image and find the most similar one from a database. Similarity is determined using cosine similarity between image embeddings, which are generated with a pretrained VGG16 model (excluding the top layer to obtain embeddings from the last hidden layer). To improve performance, embeddings are precomputed and stored in a pickle file, as computing them for each image on the fly was significantly time-consuming. The application then displays the most similar image to the user.

## How to Run
First Run the q3_database_script.py (skip if database is already downloaded in root folder of main.py)
Then run compute_embeddeds.py (can be skipped but then first run of main.py will take time to compute and store embeddeds into pkl file)
Then Run main.py using command streamlit run main.py

## Requirements
1. Tensorflow
2. Streamlit
3. pandas
4. tqdm
5. requests
