import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

# Setting dataset path and model parameters
data_path = "C:\\Users\\thefa\\OneDrive\\Desktop\\Insurance Dataset"
num_classes = 3
img_height, img_width = 200, 200
batch_size = 16

# Split dataset into train and test
train_data, test_data = train_test_split(os.listdir(data_path), test_size=0.2, random_state=42)

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode="nearest"
)

# Test data generator (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

test_generator = test_datagen.flow_from_directory(
    "C:\\Users\\thefa\\OneDrive\\Desktop\\Test Dataset",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    classes=test_data
)

## Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Training the model
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Save the model
model.save("insurance_classification_model.h5")
 
# Define the function to preprocess the input image
def preprocess_image(img_path, target_size):
    img = load_img.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

# Set the path to the image you want to classify
img_path = "C:\\Users\\thefa\\OneDrive\\Desktop\\New folder\\test"

# Preprocess the input image
img_height, img_width = 200, 200
input_image = preprocess_image(img_path, (img_height, img_width))

# Predict the class using the model
predictions = model.predict(input_image)

# Get the class with the highest probability
predicted_class_index = np.argmax(predictions[0])

# Retrieve the class label from the generator's class_indices dictionary
class_labels = list(train_generator.class_indices.keys())
predicted_class_label = class_labels[predicted_class_index]

# Print the predicted class label
print("Predicted class label:", predicted_class_label)