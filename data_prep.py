import numpy as np
import cv2
import pandas as pd
import os

IMAGE_SIZE = (331, 331)
IMAGE_FULL_SIZE = (331, 331, 3)

trainingFolder = "F:/Projects/DogBreedData/dog-breed-identification/train" # replace with your correct PATH to the training data from Kaggle

# Read CSV file
df = pd.read_csv("F:/Projects/DogBreedData/dog-breed-identification/labels.csv") # replace with your correct PATH to the training data from Kaggle
print("head of labels:")
print("==============")
print(df.head())
print(df.describe())

print("Group by labels: ")
groupByLabels = df.groupby("breed")["id"].count()
print(groupByLabels.head(10))

# Display Image

imagePath = "/Volumes/T7 Touch/Projects/DogBreedData/dog-breed-identification/train/0a001d75def0b4352ebde8d07c0850ae.jpg" # Test image
img = cv2.imread(imagePath)
# cv2.imshow("img", img)
# cv2.waitKey(0)

# Prepare images and labels as numpy array

allImages = []
allLabels = []

for index, (image_name, breed) in enumerate(df[['id', 'breed']].values):
    image_dir = os.path.join(trainingFolder, image_name + '.jpg')
    print(image_dir)

    image = cv2.imread(image_dir)
    resizedImage = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    allImages.append(resizedImage)
    allLabels.append(breed)

print(len(allLabels))
print(len(allImages))

print("save the data: ")
np.save("./temp/allImages.npy", allImages)
np.save("./temp/allLabels", allLabels)

print("Finish code")
