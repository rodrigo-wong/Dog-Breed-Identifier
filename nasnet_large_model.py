import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.nasnet import NASNetLarge

IMAGE_SIZE = (331, 331)
IMAGE_FULL_SIZE = (331, 331, 3)
batchSize = 8

allImages = np.load("./temp/allImages.npy")
allLabels = np.load("./temp/allLabels.npy")

print(allImages.shape)
print(allLabels.shape)

# Convert labels text to integers
print(allLabels)

le = LabelEncoder()
integerLabels = le.fit_transform(allLabels)
print(integerLabels)

# Unique integer labels
numOfClasses = len(np.unique(integerLabels))
print(numOfClasses)

# Convert integer labels to categorical -> prepare for training
allLabelsForModel = to_categorical(integerLabels, num_classes=numOfClasses)
print(allLabelsForModel)


# Normalize the images from 0-255 to 0-1
allImagesForModel = allImages / 255.0

print("Before split train and test data")

# Create train and test data
X_train, X_test, y_train, y_test = train_test_split(allImagesForModel, allLabelsForModel, test_size=0.3, random_state=42)

print("X_train, X_test, y_train, y_test --------> shapes: ")

print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)

# Free some memory

del allImages
del allLabels
del integerLabels
del allImagesForModel

# Build the model

myModel = NASNetLarge(input_shape=IMAGE_FULL_SIZE, weights='imagenet', include_top=False)

# Don't train the existing layers

for layer in myModel.layers:
    layer.trainable = False
    print(layer.name)

# Add Flatten Layer
plusFlattenLayer = Flatten()(myModel.output)

# Add the last dense layer without 120 classes
prediction = Dense(numOfClasses, activation='softmax')(plusFlattenLayer)

model = Model(inputs=myModel.input, outputs=prediction)

print(model.summary())

lr = 1e-4
opt = Adam(lr)

model.compile(
    loss= 'categorical_crossentropy',
    optimizer= opt,
    metrics= ['accuracy']
)

stepsPerEpoch = int(np.ceil(len(X_train) / batchSize))
validationSteps = int(np.ceil(len(X_test) / batchSize))

# Early Stopping
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

best_model_file = "temp/dogs.keras"
callbacks = [
    ModelCheckpoint(best_model_file, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor='val_accuracy', patience=7, verbose=1)
]

# Train the model (fit)
r = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=batchSize,
    steps_per_epoch=stepsPerEpoch,
    validation_steps=validationSteps,
    callbacks=[callbacks]
)
