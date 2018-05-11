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
