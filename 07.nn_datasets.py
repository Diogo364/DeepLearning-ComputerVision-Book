from nn.neuralnetwork import NeuralNetwork
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', choices=('and', 'or', 'xor'),
    required=True, help='Enter type of bitwise dataset to apply the perceptron')
ap.add_argument('-e', '--epochs', type=int, default=20000,
    help='# of epochs to fit dataset')
args = vars(ap.parse_args())


datasets = {
    'and': [[0], [0], [0], [1]],
    'or': [[0], [1], [1], [1]],
    'xor': [[0], [1], [1], [0]]
}

# construct the dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array(datasets[args['dataset']])


# define our 2-2-1 neural network and train it
print(f'[INFO] training Neural Network for {args["epochs"]} epochs')
nn = NeuralNetwork([2, 2, 1], alpha=0.5)
nn.fit(X, y, epochs=args['epochs'])


print('[INFO] testing the Neural Network')
for x, target in zip(X, y):
    # make a prediciton on the data point and display the result
    # to our console
    pred = nn.predict(x)
    step = 1 if pred > 0.5 else 0
    print(f'[INFO] data={x}, ground-truth={target[0]}, pred={pred[0][0]:.4f}, step={step}')
