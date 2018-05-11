import tensorflow as tf
from tqdm import tqdm
import numpy as np

class BaseTester:
    def __init__(self, sess, model, data, config):
        self.model = model
        self.config = config
        self.sess = sess
        self.data = data

    def epoch(self, subset):
        """
        implement the logic of epoch:
        -loop ever the number of iteration in the config and call the test step
        -add any summaries you want using the sammary
        """
        raise NotImplementedError

    def step(self, dim, subset, idx):
        """
        implement the logic of the test step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

    def pred_argmax(self,yp10, yp8, yp6, yp4, dim, priors=False):

        if priors:
            yp10 = self.norm(yp10 / self.data.priors[dim][10])
            yp8  = self.norm(yp8  / self.data.priors[dim][8 ])
            yp6  = self.norm(yp6  / self.data.priors[dim][6 ])
            yp4  = self.norm(yp4  / self.data.priors[dim][4 ])

        yp10, yp8, yp6, yp4 = np.argmax(yp10,axis=1),np.argmax(yp8,axis=1),np.argmax(yp6,axis=1),np.argmax(yp4,axis=1)

        if self.config.target_cluster == 4:
            pred = self.data.kmeans[dim][4].cluster_centers_[yp4]
        elif self.config.target_cluster == 6:
            pred = self.data.kmeans[dim][6].cluster_centers_[yp6]
        elif self.config.target_cluster == 8:
            pred = self.data.kmeans[dim][8].cluster_centers_[yp8]
        elif self.config.target_cluster == 10:
            pred = self.data.kmeans[dim][10].cluster_centers_[yp10]
        else:
            pred = np.mean([self.data.kmeans[dim][10].cluster_centers_[yp10],self.data.kmeans[dim][8].cluster_centers_[yp8],
                            self.data.kmeans[dim][6].cluster_centers_[yp6],self.data.kmeans[dim][4].cluster_centers_[yp4]],axis=0).flatten()
        return pred

    def pred_weighted_sum(self,yp10, yp8, yp6, yp4, dim, priors=False):

        if priors:
            yp10 = self.norm(yp10 / self.data.priors[dim][10])
            yp8  = self.norm(yp8  / self.data.priors[dim][8 ])
            yp6  = self.norm(yp6  / self.data.priors[dim][6 ])
            yp4  = self.norm(yp4  / self.data.priors[dim][4 ])
        
        yp10 = np.sum(yp10*self.data.kmeans[dim][10].cluster_centers_.flatten(),axis=1)
        yp8  = np.sum(yp8 *self.data.kmeans[dim][8 ].cluster_centers_.flatten(),axis=1)
        yp6  = np.sum(yp6 *self.data.kmeans[dim][6 ].cluster_centers_.flatten(),axis=1)
        yp4  = np.sum(yp4 *self.data.kmeans[dim][4 ].cluster_centers_.flatten(),axis=1)
        
        if self.config.target_cluster == 4:
            pred = yp4
        elif self.config.target_cluster == 6:
            pred = yp6
        elif self.config.target_cluster == 8:
            pred = yp8
        elif self.config.target_cluster == 10:
            pred = yp10
        else:
            pred = np.mean([yp10, yp8, yp6, yp4],axis=0).flatten()

        return pred

    def norm(self, pred):
        return pred / np.sum(pred,axis=1,keepdims=1)