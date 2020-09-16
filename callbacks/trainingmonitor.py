from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        # store the output path for the figure, the path to the JSON
        # serialized file, and starting epoch
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt
    
    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}

        # if the JSON history path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                # check to see if a starting epoch was supplied
                if self.startAt > 0:
                    # loop over the entries in the history log and
                    # trim any entries that are past the starting
                    # epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        for k, v in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l
        # check to see if the training history should be serialized
        # to file
        if self.jsonPath is not None:
            with open(self.jsonPath, 'w') as f:
                f.write(json.dumps(self.H))
        
        # ensure at least two epochs have passed before plotting
        # (epoch starts at zero)
        if len(self.H['loss']) > 1:
            # plot the training loss and accuracy
            N = np.arange(0, len(self.H['loss']))
            # plot the training loss and accuracy
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, H.history['loss'], label='train_loss')
            plt.plot(N, H.history['val_loss'], label='val_loss')
            plt.plot(N, H.history['accuracy'], label='train_acc')
            plt.plot(N, H.history['val_accuracy'], label='val_acc')
            plt.title(f'Training Loss and Accuracy [Epoch {len(self.H["loss"])}')
            plt.xlabel('Epoch #')
            plt.ylabel('Loss/Accuracy')
            plt.legend()
            plt.savefig(self.figPath)
            plt.close()