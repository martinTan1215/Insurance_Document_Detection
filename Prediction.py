import joblib
from PIL import Image
import pytesseract

# Load the saved model
clf = joblib.load("insurance_classification_model.pkl")

# Load the fitted vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Set the path to the new image you want to predict
image_path = "//Users/tanqiao/Desktop/TestDataset/Test1.png"

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
text = text.replace('Vehicle Insurance Claim Form', 'Car Insurance Claim Form')
text = text.replace('Vehicle Insurance', 'Car Insurance Claim Form')
text = text.replace('Motor Insurance', 'Car Insurance Claim Form')
text = text.replace('Property Insurance Claim Form', 'Home Insurance Claim Form')
text = text.replace('Property Insurance', 'Home Insurance Claim Form')
text = text.replace('Health Insurance', 'Health Insurance Claim Form')
text = text.replace('Life Insurance', 'Health Insurance Claim Form')

# Vectorize the preprocessed text using the fitted vectorizer
x_new = vectorizer.transform([text])

# Make the prediction
predicted_label = clf.predict(x_new)[0]
class_names = ['Home Insurance Claim Form', 'Car Insurance Claim Form', 'Health Insurance Claim Form']
predicted_class = class_names[predicted_label]

# Print the predicted class
print("Predicted class:", predicted_class)
