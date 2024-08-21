# Dog-Breed-Identifier
Description: Developed a Convolutional Neural Network (CNN) model to classify 120 dog breeds. The project involved preprocessing image data, converting labels for classification, and training the model with optimized callbacks.
GitHub Repository: https://github.com/rodrigo-wong/Dog-Breed-Identifier

Tech Stack: TensorFlow/Keras, NumPy, Scikit-learn, Panda, OpenCV

- Preprocessed images and labels to create training and testing datasets
- Leveraged NASNetLarge model for transfer learning
- Implemented callbacks for early stopping, learning rate reduction, and model checkpointing to optimize training

How to run:
1. Install all required dependencies specified in requirements.txt
2. Training data retrieved from https://www.kaggle.com/competitions/dog-breed-identification/data
3. Run data_prep.py to preprocess the data and store the generated allImages.npy and allLabels.npy files in a temp folder
4. Run nasnet_large_model to train the keras model and store in model folder
5. Run main.py with your desired dog images