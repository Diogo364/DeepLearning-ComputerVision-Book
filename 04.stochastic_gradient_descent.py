from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
    # Compute the sigmoid activation value for a given input
    return 1.0 / (1 + np.exp(-x))

def predict(X, W):
    # Take the dot product between our features weight matrix
    preds = sigmoid_activation(X.dot(W))

    # Apply a step funtion to threshold the outputs to binary
    # class labels
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1
    return preds

def next_batch(x, y, batchSize):
    # loop over our dataset 'x' in mini-batches, yelding a tuple of
    # the current batched data and labels
    for i in np.arange(0, x.shape[0], batchSize):
        yield (x[i:i+batchSize], y[i:i+batchSize])

ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', type=float, default=100,
    help='# of epochs')
ap.add_argument('-a', '--alpha', type=float, default=0.01,
    help='learning rate')
ap.add_argument('-b', '--batch-size', type=int, default=32,
    help='size of SGD mini-batches')
args = vars(ap.parse_args())


# generate a 2-class classification problem with 1,000 data proints,
# where each data point is a 2D feature vector
(X, Y) = make_blobs(n_samples=1000, n_features=2, centers=2,
    cluster_std=1.5, random_state=1)
Y = Y.reshape((-1, 1))

# Insert a column of 1's as the last entry in the feature
# matrix -- this little trick allows us to treat the bias
# as a trainable parameter within the weight matrix
X = np.c_[X, np.ones((X.shape[0]))]

# partition the data into training and testing splits using 50% of
# the data for training and the remaining 50% for testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, 
    test_size=0.5, random_state=42)


# initialize our weight matrix and list of losses
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []


# loop over the desired number of epochs
for epoch in np.arange(0, args["epochs"]):
    # initialize the total loss for the epoch
    epochLoss = []
    
    # loop over our data in batches
    for (batchX, batchY) in next_batch(X, Y, args['batch_size']):
        # take the dot product between our features ‘X‘ and the weight
        # matrix ‘W‘, then pass this value through our sigmoid activation
        # function, thereby giving us our predictions on the dataset
        preds = sigmoid_activation(batchX.dot(W))

        # now that we have our predictions, we need to determine the
        # ‘error‘, which is the difference between our predictions and
        # the true values
        error = preds - batchY
        epochLoss.append(np.sum(error ** 2))
    
        # features and the error of the predictions
        gradient = batchX.T.dot(error)

        # in the update stage, all we need to do is "nudge" the weight
        # matrix in the negative direction of the gradient (hence the
        # term "gradient descent" by taking a small step towards a set
        # of "more optimal" parameters
        W += -args["alpha"] * gradient

    # update our loss history by taking the average loss across all
    # batches
    loss = np.average(epochLoss)
    losses.append(loss)
    # check to see if an update should be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1),
            loss))


# evaluate our model
print("[INFO] evaluating...")
preds = predict(x_test, W)
print(classification_report(y_test, preds))


# plot the (testing) classification data
plt.style.use("ggplot")
fig, ax = plt.subplots(1,2, figsize=(12,5))
ax[0].set_title("Data")
ax[0].scatter(x_test[:, 0], x_test[:, 1], marker="o", c=y_test, s=30)

# construct a figure that plots the loss over time
ax[1].plot(np.arange(0, args["epochs"]), losses)
ax[1].set_title('Training Loss')
ax[1].set_xlabel('Epoch #')
ax[1].set_ylabel('Loss')
fig.savefig('./outputs/4.stochatic_gradient_descent_plot.png')


