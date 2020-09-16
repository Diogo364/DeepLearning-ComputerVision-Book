from sklearn.preprocessing import LabelBinarizer
from nn.conv.minivggnet import MiniVGGNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument('-w', '--weights', required=True,
    help='path to the best model weights file')
args = vars(ap.parse_args())


# show information on the process ID
print(f'[INFO] process ID: {os.getpid()}')

# load the training and testing data, scale it into the range [0, 1],
# then reshape the design matrix
print('[INFO] loading CIFAR-10 data...')
(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype(float) / 255.0
testX = testX.astype(float) / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
    ]

# initialize the optimizer and model
print('[INFO] compiling model...')
opt = SGD(lr=1e-2, decay=1e-2 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=opt,
    metrics=['accuracy'])

# construct the callback to save only the *best* model to disk
fname = os.path.sep.join([args['weights'], 
    'best_model.hdf5'])
checkpoint = ModelCheckpoint(fname, monitor='val_loss', mode='min',
    save_best_only=True, verbose=1)
callbacks = [checkpoint]


# train the network
print("[INFO] training the network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    batch_size=64, epochs=40, callbacks=callbacks, verbose=2)
