from nn.perceptron import Perceptron
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', choices=('and', 'or', 'xor'),
    required=True, help='Enter type of bitwise dataset to apply the perceptron')
ap.add_argument('-e', '--epochs', type=int, default=20,
    help='# of epochs to fit dataset')
args = vars(ap.parse_args())


datasets = {
    'and': [[0], [0], [0], [1]],
    'or': [[0], [1], [1], [1]],
    'xor': [[0], [1], [1], [0]]
}

# construct the OR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

y = np.array(datasets[args['dataset']])

# define our perceptron and train it
print(f'[INFO] training perceptron for {args["epochs"]} epochs')
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=args['epochs'])


print('[INFO] testing perceptron')

for x, target in zip(X, y):
    # make a prediciton on the data point and display the result
    # to our console
    pred = p.predict(x)
    print(f'[INFO] data={x}, ground-truth={target[0]}, pred={pred}')