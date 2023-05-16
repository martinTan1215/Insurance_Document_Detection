from PIL import Image
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
from sklearn.model_selection import train_test_split

# Set the path to your images
image_dir = 'C:\\Users\\thefa\\OneDrive\\Desktop\\Insurance Dataset'

# Set the names of your classes
classes = ['Home Insurance', 'Car Insurance', 'Health Insurance']

# Create lists to store the data and labels
x_data = []
y_data = []


# input image dimensions
img_rows, img_cols = 28, 28

# Loop over each class
for label, class_name in enumerate(classes):
    # Set the path to the class folder
    class_dir = os.path.join(image_dir, class_name)
    
    # Loop over each image in the class folder
    for filename in os.listdir(class_dir):
        # Open the image
        image = Image.open(os.path.join(class_dir, filename))
        
        # Preprocess the image here (e.g. resize, normalize, etc.)
        # Resize the image
        image = image.resize((img_rows, img_cols))

        # Convert the image to grayscale
        image = image.convert('L')
        
        # Convert the image to a NumPy array and add it to the data list
        x_data.append(np.array(image))
        
        # Add the label to the labels list
        y_data.append(label)

# Convert the data and labels to NumPy arrays
x_data = np.array(x_data)
y_data = np.array(y_data)


# Split the data into training and testing sets here
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

# Set parameters
batch_size = 16 # batch size for training
num_classes = 3 # number of classes (Home Insurance, Car Insurance, Health Insurance)
epochs = 12 # number of epochs to train for

# Resize the image
image = image.resize((img_rows, img_cols))

# Convert the image to grayscale
image = image.convert('L')

# Reshape the data to fit the model
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# Normalize the pixel values
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Build the model
model = Sequential()

# Add a convolutional layer
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

# Add another convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add a max pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add a dropout layer
model.add(Dropout(0.25))

# Flatten the output
model.add(Flatten())

# Add a dense layer
model.add(Dense(128, activation='relu'))

# Add another dropout layer
model.add(Dropout(0.5))

# Add the output layer
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# Save the model
model.save("insurance_classification_model.h5")