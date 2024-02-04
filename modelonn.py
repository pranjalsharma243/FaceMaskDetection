
# import the necessary packages
from tensorflow.keras.models import load_model, save_model
import argparse
import tf2onnx
import onnx

def modelonn():
# Create an argument parser to parse command line arguments
ap = argparse.ArgumentParser()

# Add argument for specifying the path to the trained face mask detection model
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to trained face mask detector model")

# Add argument for specifying the output path for the converted ONNX model
ap.add_argument("-o", "--output", type=str,
                default='mask_detector.onnx',
                help="path to output the face mask detector model in ONNX format")

# Parse the arguments provided by the user
args = vars(ap.parse_args())

# Load the trained face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model(args["model"])

# Convert the loaded Keras model to the ONNX format with an opset version of 13
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)

# Modify the input and output dimensions of the ONNX model to be dynamic
onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = '?'

# Save the converted ONNX model to the specified output path
onnx.save(onnx_model, args['output'])

if __name__ == "__main__":
	modelonn()
