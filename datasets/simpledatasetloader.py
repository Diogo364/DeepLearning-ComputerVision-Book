import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        '''
        Store the image preprocessor
        '''
        self.preprocessors = preprocessors if preprocessors is not None else []


    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []
        set_path = set()

        for idx, imagePath in enumerate(imagePaths):
            '''
            Load the image and extract the class label assuming
            that our path has the following format:
            /path/to/dataset/{class}/{image}.jpg
            '''
            
            image = cv2.imread(imagePath)
            set_path.add(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            '''
            Apply preprocessors into image
            '''
            if self.preprocessors is not None:
                for preprocess in self.preprocessors:
                    image = preprocess.preprocess(image)
            '''
            treat our processed image as a "feature vector" 
            by updating the data list followed by the labels
            '''
            data.append(image)
            labels.append(label)

             # show an update every ‘verbose‘ images
            if verbose > 0 and idx > 0 and (idx + 1) % verbose == 0: 
                print(f"[INFO] processed {idx+1}/{len(imagePaths)}")
    
        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))