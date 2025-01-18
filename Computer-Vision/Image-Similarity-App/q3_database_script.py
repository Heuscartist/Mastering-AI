import os
import pandas as pd
import requests
from tqdm import tqdm

# This script is used to download images from a CSV file that contains links to various images
# Then it stores the images into the specified folder


# Create a folder to save the images
output_folder = "q3_database"
os.makedirs(output_folder, exist_ok=True)

# Load the CSV file which is currently in the root directory
csv_file = "q3_dataset.csv"
data = pd.read_csv(csv_file)

# Ensuring that the correct CSV file is loaded by checking for the required columns
if "Product ID" not in data.columns or "image_link" not in data.columns:
    raise ValueError("CSV file must contain 'Product ID' and 'image_link' columns.")

# Download images by extractig the Product ID and image link from the CSV file
for index, row in tqdm(data.iterrows(), total=len(data)):
    product_id = row["Product ID"]
    image_link = row["image_link"]

    try:
        # Get the image data using the requests libray
        response = requests.get(image_link, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        # Save the image to the output folder according to the name of the product ID
        file_extension = image_link.split('.')[-1].split('?')[0]  # Get file extension
        file_name = f"{product_id}.{file_extension}"  # Save as ProductID.extension
        file_path = os.path.join(output_folder, file_name)

        with open(file_path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {image_link}: {e}")

print(f"Images downloaded successfully into '{output_folder}' folder.")
