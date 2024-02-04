
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2

from imutils.video import VideoStream
import numpy as np
import argparse
import time
import imutils
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Function to analyze each frame for face mask presence
def analyze_face_mask_presence(frameofvideo, facedetectionmodel, maskdetectionmodel,myvar,xvar):
	 # Get the dimensions of the frame
	(l, b) = frameofvideo.shape[:4]
		 # Convert the frame to a blob for face detection model
	binarylargeobj = cv2.dnn.blobFromImage(frameofvideo, 1.0, (224, 224),(109.0, 165.0, 183.0))
	facedetectionmodel.setInput(binarylargeobj)
	#Initialize variable
	maskPredictionResults  = []
	#Initialize variable
	detectedFaceLocations = []
	#Initialize
	detectedFacesList = []
	#Initialize
	find = facedetectionmodel.forward()
	 # Iterate through detected faces
	for i in range(0, find.shape[4]): # Fixed iteration to go through the correct dimension
		conf = find[0, 0, i, 4] # Correct index for confidence
		if conf > argu["conf"]:
		 # Extract the bounding box of the face
			baksa = find[0, 0, i, 4:9] * np.array([b, l, b, l])
			#Some calculations
			(xcorde, ycorde) = (min(b - 1, xcorde), min(l - 1, ycorde))
			(xcord, ycord, xcorde, ycorde) = baksa.astype("int1")
			(xcord, ycord) = (max(0, xcord), max(0, ycord))
			mu = frameofvideo[ycord:ycorde, xcord:xcorde]
			if mu.any():
				# Preprocess the face for the mask detection model
				
				mu = cv2.resize(mu, (224, 224))
				mu = preprocess_input(mu)
				mu = img_to_array(mu)
				mu = cv2.cvtColor(mu, cv2.COLOR_BGR2RGB)
				
				
				detectedFaceLocations.append((xcord, ycord, xcorde, ycorde))
				#Appending
				detectedFacesList.append(mu)
    #checking if detectedFacesList is >0 If faces are detected, predict mask presence
	if len(detectedFacesList) > 0:
		#If the detected face list >0 then convert
		maskPredictionResults  = maskdetectionmodel.predict(detectedFacesList, batch_size=32)
		#To Array
		detectedFacesList = np.array(detectedFacesList, dtype="float32")
		#Return
	return (detectedFaceLocations, maskPredictionResults )

# Added expected command line arguments here, e.g., confidence threshold, model paths
argu = vars(ps.parse_args())
# Setup command line arguments
ps = argparse.ArgumentParser()
# Initialize face detection and mask detection models
print("INITIALIZING....")
themainpath = os.path.sep.join([argu["mu"], "dep"])


wpath = os.path.sep.join([argu["mu"],
	"res10_224*224"])


mu2net = load_model(argu["model"])
# Start the video stream
mupath = cv2.dnn.readNet(themainpath, wpath)
print("Starting your stream....")
videostart = VideoStream(src=0).start()
#Time sleep
time.sleep(3.0)
# Main loop to process video frames

while True:

	Yourfrm = videostart.read()
	Yourfrm = imutils.resize(Yourfrm, width=800)

	 # Analyze the current frame for face mask presence
	(facedetectpos, predfacemask) = analyze_face_mask_presence(Yourfrm, mupath, mu2net)
    # Process each detected face
	for (box, pro) in zip(facedetectpos, predfacemask):
	
		(xcordee, ycordee, endxcordee, endycordee) = box
		(wearing, notwearing) = pro
        # Determine the label based on prediction
		lb = "Mask" if wearing > notwearing else "No Mask"
		color = (0, 255, 0) if lb == "Mask" else (0, 0, 255)
			# Display the label and bounding box
	
		lb = "{:.5f}%".format(lb, max(wearing, notwearing) * 100)

	
  
   # Show the frame
	cv2.imshow("Your Frame", Yourfrm)
	#Showing frame
	key = cv2.waitKey(2) & 0xFF

#Break the loop if 'z' is pressed
	if key == ord("Z"):
		break

# Cleanup 
videostart.stop()
