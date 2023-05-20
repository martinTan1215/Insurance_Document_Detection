import joblib
from PIL import Image
import pytesseract
import os
import shutil


# Load the saved model
clf = joblib.load("insurance_classification_model.pkl")

# Load the fitted vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Set the path to the folder containing the images to predict
folder_path = "/Users/tanqiao/Desktop/Test_Dataset"

# Set the paths for the output folders
output_folder_home = "/Users/tanqiao/Desktop/DestinationFolder/Home"
output_folder_car = "/Users/tanqiao/Desktop/DestinationFolder/Car"
output_folder_health = "/Users/tanqiao/Desktop/DestinationFolder/Health"

# Create the output folders if they don't exist
os.makedirs(output_folder_home, exist_ok=True)
os.makedirs(output_folder_car, exist_ok=True)
os.makedirs(output_folder_health, exist_ok=True)

# Iterate over the images in the folder
for file_name in os.listdir(folder_path):
    # Construct the full path to the image
    image_path = os.path.join(folder_path, file_name)

    # Open the image
    image = Image.open(image_path)

    # Resize the image pixels
    image = image.resize((500, 500))

    # Convert the image to grayscale
    image = image.convert('L')

    # Use pytesseract to convert the image to text
    text = pytesseract.image_to_string(image)

    # Preprocess the text
    text = text.replace('Auto Insurance Claim Form', 'Car Insurance Claim Form')
    text = text.replace('Auto Insurance', 'Car Insurance Claim Form')
    text = text.replace('Vehicle Insurance Claim Form',
                        'Car Insurance Claim Form')
    text = text.replace('Vehicle Insurance', 'Car Insurance Claim Form')
    text = text.replace('Motor Insurance', 'Car Insurance Claim Form')
    text = text.replace('Property Insurance Claim Form',
                        'Home Insurance Claim Form')
    text = text.replace('Property Insurance', 'Home Insurance Claim Form')
    text = text.replace('Health Insurance', 'Health Insurance Claim Form')
    text = text.replace('Life Insurance', 'Health Insurance Claim Form')

    # Vectorize the preprocessed text using the fitted vectorizer
    x_new = vectorizer.transform([text])

    # Make the prediction
    predicted_label = clf.predict(x_new)[0]
    class_names = ['Home Insurance Claim Form', 'Car Insurance Claim Form',
                   'Health Insurance Claim Form']
    predicted_class = class_names[predicted_label]

    # Define the destination folder based on the predicted class
    if predicted_class == 'Home Insurance Claim Form':
        destination_folder = output_folder_home
    elif predicted_class == 'Car Insurance Claim Form':
        destination_folder = output_folder_car
    elif predicted_class == 'Health Insurance Claim Form':
        destination_folder = output_folder_health

    # Copy the image to the corresponding destination folder
    shutil.copy(image_path, os.path.join(destination_folder, file_name))

    # Print the predicted class and destination folder for current image
    print(f"Predicted class for {file_name}: {predicted_class}")
    print(f"Destination folder: {destination_folder}")
