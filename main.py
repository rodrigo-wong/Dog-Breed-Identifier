import numpy as np
import tensorflow as tf
import cv2

# Load the trained model
modelFile = "./model/dogs.keras"
model = tf.keras.models.load_model(modelFile)

# Define input shape for the model
inputShape = (331, 331)

# Load the labels (categories)
allLabels = np.load("./temp/allLabels.npy")
categories = np.unique(allLabels)
# print(categories)

def prepareImage(img):
    resized = cv2.resize(img, inputShape, interpolation=cv2.INTER_AREA)
    imgResult = np.expand_dims(resized, axis=0)
    imgResult = imgResult / 255.0  # Normalize the image
    return imgResult

# Path to the test image
testImagePath = "./test/italian_greyhound_test.jpg"

# Read and prepare the image for the model
img = cv2.imread(testImagePath)
imageForModel = prepareImage(img)

# Predict the class of the image
resultArray = model.predict(imageForModel)
answers = np.argmax(resultArray, axis=1)

# Get the corresponding label from the categories
predicted_breed = categories[answers[0]]

# Get the accuracy (confidence level) of the prediction
confidence = resultArray[0][answers[0]]

print(f"Predicted breed: {predicted_breed} with confidence: {confidence:.2%}")
