from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
    help='path to input dataset')
ap.add_argument('-m', '--model', required=True,
    help='path to output model')
args = vars(ap.parse_args())

classLabels = ['cat', 'dog', 'panda']


# grab the list of images in the dataset then randomly sample
# indexes into the image paths list
print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]


# initialize the image preporcessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
data, labels = sdl.load(imagePaths)
data = data.astype(float) / 255.0

# load the pre-trained network
print('[INFO] loading pre-trained network')
model = load_model(f'{args["model"]}/14.shallownet_weights.hdf5')

# making predictions
print("[INFO] predicting...")
preds = model.predict(data, batch_size=32).argmax(axis=1)


# loop over the sample images
for (i, imagePath) in enumerate(imagePaths):
    # load the example image, draw the prediction, and display it
    # to our screen
    image = cv2.imread(imagePath)
    cv2.putText(image, f"Label: {classLabels[preds[i]]}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(f"./outputs/15.{i}-image.png", image)