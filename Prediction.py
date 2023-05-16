from PIL import Image
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
from sklearn.model_selection import train_test_split
from keras.models import load_model

# Load the trained model
model = load_model("insurance_classification_model.h5")

# input image dimensions
img_rows, img_cols = 28, 28

# Set the names of your classes
classes = ['Home Insurance', 'Car Insurance', 'Health Insurance']

def predict_image(image_path):
    # Open the image
    image = Image.open(image_path)

    # Resize and convert the image to grayscale
    image = image.resize((img_rows, img_cols)).convert('L')

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Reshape and normalize the image
    image_array = image_array.reshape(1, img_rows, img_cols, 1).astype('float32') / 255

    # Make a prediction using the model
    prediction = model.predict(image_array)

    # Find the class that has the highest probability
    predicted_class = np.argmax(prediction)

    # Print the class name
    print(f'The model predicts that the image is: {classes[predicted_class]}')

# Test the function with an image of your choice
predict_image('C:\\Users\\thefa\\OneDrive\\Desktop\\Insurance Dataset\\Car Insurance\\car-insurance-claim-form-1.jpg')
