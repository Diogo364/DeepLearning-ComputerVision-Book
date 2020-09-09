from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
    help='path to input dataset')
args = vars(ap.parse_args())

# grab the list of image paths
print('[INFO] loading images...')
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image prepocessor, load the dataset from disk,
# and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
data, labels = sdl.load(imagePaths, verbose=500)
data = data.reshape(data.shape[0], 3072)

le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
trainX, testX, trainY, testY = train_test_split(data, labels,
    train_size=0.75, random_state=5)

# loop over our set of regularizers
for r in (None, 'l1', 'l2'):
    # train a SGD classifier using a softmax loss function and the
    # specified regularization function for 10 epochs
    print(f'[INFO] training model with `{r} penalty`')
    model = SGDClassifier(loss='log', penalty=r, max_iter=10,
        learning_rate='constant', eta0=0.01, random_state=42)
    model.fit(trainX, trainY)
    
    # evaluate the classifier
    acc = model.score(testX, testY)
    print(f'[INFO] `{r}` penalty accuracy: {acc*100:.2f}%')