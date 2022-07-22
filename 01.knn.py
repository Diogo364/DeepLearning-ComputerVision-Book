from enum import auto
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse
import mlflow

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True, 
        help='path to input dataset')
    ap.add_argument('-k', '--neighbors', type=int, default=1, 
        help='# of nearest neighbors for classification')
    ap.add_argument('-j', '--jobs', type=int, default=-1, 
        help='# of jobs for K-NN distance (-1 uses all available cores)')
    return vars(ap.parse_args())


if __name__ == '__main__':
    mlflow.sklearn.autolog()
    args = get_args()
    
    # grab the list of images that we’ll be describing
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args["dataset"]))
    # initialize the image preprocessor, load the dataset from disk,
    # and reshape the data matrix
    sp = SimplePreprocessor(32, 32)
    sdl = SimpleDatasetLoader(preprocessors=[sp])
    (data, labels) = sdl.load(imagePaths, verbose=500)
    data = data.reshape((data.shape[0], 3072))
    # show some information on memory consumption of the images
    print(f"[INFO] features matrix: {data.nbytes/(1024 * 1000.0):.1f}MB")

    with mlflow.start_run(run_name='knn-image-classifier') as run:
        le = LabelEncoder()
        labels = le.fit_transform(labels)

        x_train,  x_test, y_train, y_test = train_test_split(data, labels,
            test_size=0.25, random_state=42)

        print("[INFO] evaluating k-NN classifier...")
        knn = KNeighborsClassifier(n_neighbors=args['neighbors'],
            n_jobs=args['jobs'])

        knn.fit(x_train, y_train)

        print(classification_report(y_test, knn.predict(x_test), target_names=le.classes_))
