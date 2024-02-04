from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import argparse
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt

#Getting the path to the dataset
argp = argparse.ArgumentParser()
argp.add_argument("-d", "--ds", required=True,
	help="This is the path to the dataset")

args = vars(argp.parse_args())
#Path to the output face model
argp.add_argument("-m", "--model1", type=str,default="maskdetector.model",
	help="This is the path to output model")

print("Initializing the images")
#Getting the path of the image
pathoftheimage = list(paths.list_images(args["dataset"]))
#initializing data variable
data = []
#initializing labels array
lbs = []

BASE_LR = 1e-9
EP = 15
BS = 40

#Iterating over the images
for pathofimage in pathoftheimage:
	lb = pathofimage.split(os.path.sep)[-4]
	#Specifying the size along with the path
	img = load_img(pathofimage, target_size=(224, 224))
	img = img_to_array(img)
	img = preprocess_input(img)
	data.append(img)
	#appending to the labels
	lbs.append(lb)

#Converting the data into the array of NumPy 
data = np.array(data, dtype="float32")
#Appending
lbs = np.array(lbs)

# Performing the encoding
lb = LabelBinarizer()
lbs = lb.fit_transform(lbs)
lbs = to_categorical(lbs)

#80% of the data is used for training and 20% of the data is used for testing so we are spliting
#Splitting the data
(xtrain, xtest, ytrain, ytest) = train_test_split(data, lbs,
	test_size=0.20, stratify=lbs, random_state=42)

# loading MobileNetV2, It ensures the head Fully Connected layer sets are
# left off
baseModel = MobileNetV2(weights="imagen", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# In this we have done the data augmentation
aug = ImageDataGenerator(
	#Rotating the image 15%
	rotation_range=15,
	#Zooming the image 20%
	zoom_range=0.20,
	#Changing other attributes
	width_shift_range=0.1,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")


#Here we have constructed a model on top of the base model
HM = baseModel.output
#We have various layers as we discussed in the second Presentation 6 pooling layers
HM = AveragePooling2D(pool_size=(6, 6))(HM)
HM = Flatten(name="flatten")(HM)
HM = Dense(128, activation="relu")(HM)
HM = Dropout(0.5)(HM)
HM = Dense(2, activation="softmax")(HM)

# Here we are placing the head  model on top of the base model 
# The actual model we are going to train
model = Model(inputs=baseModel.input, outputs=HM)

#Looping over all the layers
for la in baseModel.layers:
	la.trainable = False

# Here we compile  thats how we fin loss/accuracy using Adam function
pt = Adam(lr=BASE_LR, decay=BASE_LR / EPOCHS)
#Compile
model.compile(loss="ce", optimizer=pt,
	metrics=["accuracy"])

# Here we train the head of the model
H = model.fit(
	aug.flow(xtrain, ytrain, batch_size=BS),
	#Finding steps per epoch
	steps_per_epoch=len(xtrain) // BS,
	validation_data=(xtest, ytest),
	validation_steps=len(xtest) // BS,
	#Finding epochs
	epochs=EPOCHS)

# Here we make predictions on testing set
predictit = model.predict(xtest, batch_size=BS)

#Identify the index of the label with the highest predicted probability for each image in the test set
predictit = np.argmax(predictit, axis=1)

#Here we have classification report
print(classification_report(ytest.argmax(axis=1), predictit,
	target_names=lb.classes_))

#Serialize the model, saving to the disk
model.save(args["model"], save_format="h5")

#Plotting the loss/Accuracy graph
N = EPOCHS
plt.style.use("ggplot")
#Plotting the figure
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="Training Loss")
#Validation Loss
plt.plot(np.arange(0, N), H.history["Validation Loss"], label="Validation Loss")
#Accuracy
plt.plot(np.arange(0, N), H.history["Accuracy"], label="Training Accuracy")
#Validation Accuracy
plt.plot(np.arange(0, N), H.history["Validation Accuracy"], label="Validation Accuracy")
#Xlabel
plt.xlabel("Epoch #")
#title
plt.title("Training Loss, Accuracy Graph")
#ylabel
plt.ylabel("Loss, Accuracy")

