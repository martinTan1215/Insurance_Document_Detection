import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

# Define the input shape of the images
img_height = 224
img_width = 224
channels = 3
input_shape = (img_height, img_width, channels)

# Define the number of classes (categories) for the images
num_classes = 5

# Create a sequential model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

# Add dense layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# Data augmentation for the training images
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_height, img_width), batch_size=32, class_mode='categorical')

# Data augmentation for the validation images
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(img_height, img_width), batch_size=32, class_mode='categorical')

# Train the model
model.fit_generator(train_generator, steps_per_epoch=100, epochs=10, validation_data=val_generator, validation_steps=50)

# Evaluate the model on the test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_height, img_width), batch_size=32, class_mode='categorical', shuffle=False)
score = model.evaluate_generator(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Make predictions on new images
test_image = keras.preprocessing.image.load_img('/path/to/test/image.jpg', target_size=(img_height, img_width))
test_image = keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0
predicted_class = np.argmax(model.predict(test_image), axis=-1)
class_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
print('Predicted class:', class_names[predicted_class[0]])
